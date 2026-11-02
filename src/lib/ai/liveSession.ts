/** Opt-in live transcription with durable browser-side PCM queue and reconnect. */
import type { AiTone, TranscriptSegment } from "./contracts";
import { HandyAiClient } from "./client";
import { normalizeServerUrl } from "./serverConfig";
import { LiveAudioCapture } from "./liveAudio";

const DB_NAME = "handy-live-queue-v1";
const SESSION_ID_KEY = "handy.live.sessionId.v1";
const CHUNKS = "chunks";
const RESULTS = "results";

type Chunk = { id: string; session: string; sequence: number; pcm: ArrayBuffer };
export type LiveResult = {
  type: "final";
  sequence: number;
  segment: TranscriptSegment;
  language: string | null;
  warnings: string[];
};
type StoredResult = LiveResult & { id: string; session: string };

function dbOpen(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      for (const name of [CHUNKS, RESULTS]) {
        const store = db.createObjectStore(name, { keyPath: "id" });
        store.createIndex("session_sequence", ["session", "sequence"]);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function transact<T>(
  storeName: string,
  mode: IDBTransactionMode,
  op: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await dbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, mode);
    const request = op(tx.objectStore(storeName));
    let value: T;
    request.onsuccess = () => { value = request.result; };
    tx.oncomplete = () => { db.close(); resolve(value); };
    tx.onerror = () => { db.close(); reject(tx.error); };
    tx.onabort = () => { db.close(); reject(tx.error); };
  });
}

async function oldestPending(session: string): Promise<Chunk | undefined> {
  const db = await dbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS, "readonly");
    const index = tx.objectStore(CHUNKS).index("session_sequence");
    const range = IDBKeyRange.bound([session, 0], [session, Number.MAX_SAFE_INTEGER]);
    const cursor = index.openCursor(range, "next");
    let result: Chunk | undefined;
    cursor.onsuccess = () => { result = cursor.result?.value as Chunk | undefined; };
    tx.oncomplete = () => { db.close(); resolve(result); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

async function lastSequence(session: string): Promise<number> {
  const db = await dbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS, "readonly");
    const index = tx.objectStore(CHUNKS).index("session_sequence");
    const range = IDBKeyRange.bound([session, 0], [session, Number.MAX_SAFE_INTEGER]);
    const cursor = index.openCursor(range, "prev");
    let result = -1;
    cursor.onsuccess = () => { if (cursor.result) result = (cursor.result.value as Chunk).sequence; };
    tx.oncomplete = () => { db.close(); resolve(result); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export function loadLastLiveSession(): string | null {
  try {
    const value = localStorage.getItem(SESSION_ID_KEY);
    return value && /^[0-9a-f]{32}$/.test(value) ? value : null;
  } catch {
    return null;
  }
}

type Callbacks = {
  state: (state: string) => void;
  result: (result: LiveResult) => void;
  partial: (text: string) => void;
  error: (message: string) => void;
};

export class LiveStreamSession {
  private readonly client: HandyAiClient;
  private readonly origin: string;
  private readonly token: string;
  private readonly language: string | null;
  private readonly tone: AiTone;
  private readonly cleanup: boolean;
  private readonly callbacks: Callbacks;
  private socket: WebSocket | null = null;
  private capture: LiveAudioCapture | null = null;
  private session: string | null = null;
  private stopped = false;
  private disposed = false;
  private ready = false;
  private awaiting: number | null = null;
  private next = 0;
  private retries = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private persistedWrite: Promise<void> = Promise.resolve();
  private finishingSent = false;

  constructor(
    baseUrl: string,
    token: string,
    config: { language?: string; tone: AiTone; cleanup: boolean },
    callbacks: Callbacks,
  ) {
    this.origin = normalizeServerUrl(baseUrl);
    this.client = new HandyAiClient(baseUrl, token);
    this.token = token;
    this.language = config.language || null;
    this.tone = config.tone;
    this.cleanup = config.cleanup;
    this.callbacks = callbacks;
  }

  get sessionId(): string | null { return this.session; }

  async start(resumeSession?: string): Promise<string> {
    this.session = resumeSession || (await this.client.createSession({
      mode: "dictation", tone: this.tone, language: this.language,
    })).id;
    localStorage.setItem(SESSION_ID_KEY, this.session);
    this.next = (await lastSequence(this.session)) + 1;
    if (resumeSession) {
      const old = await this.client.getSession(resumeSession);
      if (old.status === "completed") throw new Error("Session is already completed");
    }
    this.connect();
    const capture = new LiveAudioCapture((audio) => this.enqueue(audio));
    this.capture = capture;
    try {
      await capture.start();
    } catch (error) {
      this.dispose();
      throw error;
    }
    this.callbacks.state("recording");
    return this.session;
  }

  private enqueue(pcm: ArrayBuffer): void {
    if (this.disposed || !this.session) return;
    const sequence = this.next++;
    const session = this.session;
    this.persistedWrite = this.persistedWrite.then(async () => {
      await transact(CHUNKS, "readwrite", (store) => store.put({
        id: session + ":" + sequence, session, sequence, pcm,
      } satisfies Chunk));
      await this.pump();
    }).catch((error) => this.callbacks.error(String(error)));
  }

  private connect(): void {
    if (!this.session || this.disposed) return;
    const uri = this.origin.replace(/^http/, "ws") + "/v1/stream/" + this.session;
    const ws = new WebSocket(uri);
    this.socket = ws;
    this.ready = false;
    this.awaiting = null;
    ws.onopen = () => {
      // Never include API token in URL, localStorage or logs.
      ws.send(JSON.stringify({
        type: "hello", token: this.token, language: this.language,
        cleanup: this.cleanup, tone: this.tone, last_seen: -1,
      }));
    };
    ws.onmessage = (event: MessageEvent) => { void this.receive(event.data); };
    ws.onclose = (event) => {
      this.ready = false;
      this.awaiting = null;
      if (this.disposed || event.code === 1000) return;
      if ([4400, 4401, 4403, 4404, 4409].includes(event.code)) {
        this.callbacks.error("Stream denied by the server (code " + event.code + ")");
        return;
      }
      this.callbacks.state("reconnecting");
      const backoff = Math.min(30000, 750 * 2 ** this.retries++);
      this.reconnectTimer = setTimeout(() => this.connect(), backoff);
    };
  }

  private async receive(raw: string): Promise<void> {
    if (this.disposed) return;
    let event: Record<string, unknown>;
    try { event = JSON.parse(raw) as Record<string, unknown>; }
    catch { this.callbacks.error("Malformed server response"); return; }
    switch (event.type) {
      case "ready": {
        this.ready = true;
        this.retries = 0;
        const expected = event.next_sequence;
        if (typeof expected === "number") {
          this.next = Math.max(this.next, expected, (await lastSequence(this.session!)) + 1);
        }
        for (const result of (event.results as LiveResult[] | undefined) ?? []) {
          await this.saveResult(result);
        }
        this.callbacks.state(this.stopped ? "draining" : "recording");
        await this.pump();
        break;
      }
      case "ack":
        // Raw audio is kept in IndexedDB until the final commit, not until ACK.
        break;
      case "partial":
        if (typeof event.raw_text === "string") this.callbacks.partial(event.raw_text);
        break;
      case "final": {
        const result = event as unknown as LiveResult;
        await this.saveResult(result);
        await transact(CHUNKS, "readwrite", (store) => store.delete(
          this.session + ":" + result.sequence,
        ));
        this.awaiting = null;
        this.callbacks.partial("");
        await this.pump();
        break;
      }
      case "resync":
        this.awaiting = null;
        await this.pump(event.next_sequence as number);
        break;
      case "complete":
        this.callbacks.state("completed");
        localStorage.removeItem(SESSION_ID_KEY);
        this.dispose();
        break;
      case "error":
        this.callbacks.error(String(event.reason || "Stream error"));
        break;
    }
  }

  private async saveResult(result: LiveResult): Promise<void> {
    if (!this.session) return;
    await transact(RESULTS, "readwrite", (store) => store.put({
      ...result, id: this.session + ":" + result.sequence, session: this.session,
    } satisfies StoredResult));
    this.callbacks.result(result);
  }

  private async pump(expected?: number): Promise<void> {
    if (!this.session || !this.ready || this.awaiting !== null || this.finishingSent) return;
    const ws = this.socket;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const first = await oldestPending(this.session);
    if (first) {
      if (typeof expected === "number" && first.sequence !== expected) {
        this.callbacks.error("A required audio chunk is missing from the local queue");
        return;
      }
      this.awaiting = first.sequence;
      ws.send(JSON.stringify({
        type: "chunk", sequence: first.sequence,
        encoding: "pcm_s16le", sample_rate: 16000,
      }));
      ws.send(first.pcm);
    } else if (this.stopped) {
      this.finishingSent = true;
      ws.send(JSON.stringify({ type: "finish" }));
    }
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    await this.capture?.stop();
    this.callbacks.state("draining");
    await this.persistedWrite;
    await this.pump();
  }

  dispose(): void {
    this.disposed = true;
    clearTimeout(this.reconnectTimer);
    void this.capture?.stop(false);
    this.socket?.close();
    this.ready = false;
  }
}
