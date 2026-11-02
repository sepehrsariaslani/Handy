import type {
  AiCapabilities,
  AiCleanupRequest,
  AiCleanupResponse,
  AiModelStatus,
  AiMemoryQuery,
  AiMemoryResponse,
  AiNote,
  AiNotesPage,
  AiSession,
  AiSessionCreate,
  AiTone,
  AiTranscriptionResponse,
} from "./contracts";
import { normalizeServerUrl } from "./serverConfig";

/** Authenticated phase-2 API client. Never stores or logs the API token. */
export class HandyAiClient {
  private readonly url: string;
  constructor(
    baseUrl: string,
    private readonly apiToken: string,
  ) {
    this.url = normalizeServerUrl(baseUrl);
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(this.url + path, {
      ...init,
      headers: {
        Authorization: "Bearer " + this.apiToken,
        ...init?.headers,
      },
    });
    if (!response.ok) {
      // Never print server bodies: they may contain private transcripts.
      throw new Error(
        "AI server request failed (HTTP " + response.status + ")",
      );
    }
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  capabilities(): Promise<AiCapabilities> {
    return this.request<AiCapabilities>("/v1/capabilities");
  }

  modelStatus(): Promise<AiModelStatus> {
    return this.request<AiModelStatus>("/v1/models/status");
  }

  streamState(
    id: string,
    afterSequence = -1,
  ): Promise<{
    next_sequence: number;
    pending_sequence: number | null;
    committed_count: number;
    results: Array<{
      type: "final";
      sequence: number;
      segment: {
        raw_text: string;
        clean_text: string | null;
        id: string;
        start: number;
        end: number;
        speaker: string | null;
        final: boolean;
      };
      language: string | null;
      warnings: string[];
    }>;
  }> {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid session ID");
    return this.request(
      "/v1/sessions/" + id + "/stream-state?after_sequence=" + afterSequence,
    );
  }

  cleanupText(body: AiCleanupRequest): Promise<AiCleanupResponse> {
    return this.request<AiCleanupResponse>("/v1/text/cleanup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  transcribeFile(
    audio: File,
    options: { language?: string; cleanup?: boolean; tone?: AiTone } = {},
  ): Promise<AiTranscriptionResponse> {
    const form = new FormData();
    form.append("audio", audio, audio.name);
    if (options.language) form.append("language", options.language);
    if (options.cleanup) form.append("cleanup", "true");
    if (options.tone) form.append("tone", options.tone);
    return this.request<AiTranscriptionResponse>("/v1/transcriptions", {
      method: "POST",
      body: form,
    });
  }

  createSession(body: AiSessionCreate): Promise<AiSession> {
    return this.request<AiSession>("/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  getSession(id: string): Promise<AiSession> {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid session ID");
    return this.request<AiSession>("/v1/sessions/" + id);
  }
  listNotes(query = "", offset = 0, limit = 25): Promise<AiNotesPage> {
    const params = new URLSearchParams({
      q: query,
      offset: String(offset),
      limit: String(limit),
    });
    return this.request<AiNotesPage>("/v1/notes?" + params.toString());
  }

  getNote(id: string): Promise<AiNote> {
    if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid note ID");
    return this.request<AiNote>("/v1/notes/" + id);
  }

  createNote(title: string, content: string): Promise<AiNote> {
    return this.request<AiNote>("/v1/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content }),
    });
  }

  updateNote(note: AiNote, title: string, content: string): Promise<AiNote> {
    return this.request<AiNote>("/v1/notes/" + note.id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content, version: note.version }),
    });
  }

  analyzeNote(id: string): Promise<AiNote> {
    if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid note ID");
    return this.request<AiNote>("/v1/notes/" + id + "/analyze", {
      method: "POST",
    });
  }

  importMeetingNote(jobId: string): Promise<AiNote> {
    if (!/^[0-9a-f]{32}$/.test(jobId)) throw new Error("Invalid meeting ID");
    return this.request<AiNote>("/v1/notes/from-meeting/" + jobId, {
      method: "POST",
    });
  }

  deleteNote(id: string): Promise<void> {
    if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid note ID");
    return this.request<void>("/v1/notes/" + id, { method: "DELETE" });
  }

  searchMemory(body: AiMemoryQuery): Promise<AiMemoryResponse> {
    return this.request<AiMemoryResponse>("/v1/memory/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  askMemory(body: AiMemoryQuery): Promise<AiMemoryResponse> {
    return this.request<AiMemoryResponse>("/v1/memory/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  listTasks(status?: import("./contracts").AiTaskStatus): Promise<import("./contracts").AiTaskPage> {
    const params = status ? "?status=" + encodeURIComponent(status) : "";
    return this.request("/v1/tasks" + params);
  }

  createTask(input: import("./contracts").AiTaskInput): Promise<import("./contracts").AiTask> {
    return this.request("/v1/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  }

  updateTask(
    task: import("./contracts").AiTask,
    input: import("./contracts").AiTaskInput,
    status: import("./contracts").AiTaskStatus,
  ): Promise<import("./contracts").AiTask> {
    return this.request("/v1/tasks/" + task.id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...input, status, version: task.version }),
    });
  }

  deleteTask(id: string): Promise<void> {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid task ID");
    return this.request("/v1/tasks/" + id, { method: "DELETE" });
  }

  importMeetingTasks(jobId: string): Promise<{ created: number; tasks: import("./contracts").AiTask[] }> {
    if (!/^[a-f0-9]{32}$/.test(jobId)) throw new Error("Invalid meeting ID");
    return this.request("/v1/tasks/from-meeting/" + jobId, { method: "POST" });
  }

  listConversations(): Promise<{ items: import("./contracts").AiConversation[] }> {
    return this.request("/v1/memory/conversations");
  }

  createConversation(title: string): Promise<import("./contracts").AiConversation> {
    return this.request("/v1/memory/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
  }

  getConversation(id: string): Promise<import("./contracts").AiConversationDetail> {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid conversation ID");
    return this.request("/v1/memory/conversations/" + id);
  }

  deleteConversation(id: string): Promise<void> {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new Error("Invalid conversation ID");
    return this.request("/v1/memory/conversations/" + id, { method: "DELETE" });
  }

  askInConversation(
    conversation: import("./contracts").AiConversation,
    body: import("./contracts").AiMemoryQuery,
  ): Promise<{ conversation: import("./contracts").AiConversation; result: import("./contracts").AiMemoryResponse }> {
    return this.request("/v1/memory/conversations/" + conversation.id + "/turns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, expected_turns: conversation.turn_count }),
    });
  }

  getSemanticStatus(): Promise<import("./contracts").AiSemanticStatus> {
    return this.request("/v1/memory/semantic/status");
  }

  syncSemantic(limit = 100): Promise<{ indexed: number; model: string }> {
    return this.request("/v1/memory/semantic/sync?limit=" + limit, { method: "POST" });
  }

}
