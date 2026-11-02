import type {
  AiCapabilities,
  AiCleanupRequest,
  AiCleanupResponse,
  AiModelStatus,
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
    return (await response.json()) as T;
  }

  capabilities(): Promise<AiCapabilities> {
    return this.request<AiCapabilities>("/v1/capabilities");
  }

  modelStatus(): Promise<AiModelStatus> {
    return this.request<AiModelStatus>("/v1/models/status");
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
}
