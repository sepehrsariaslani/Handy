import type { AiCapabilities, AiSession, AiSessionCreate } from "./contracts";
import { normalizeServerUrl } from "./serverConfig";

/** Phase-1 client for common authenticated session APIs. Token is memory-only. */
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
        Authorization: `Bearer ${this.apiToken}`,
        ...init?.headers,
      },
    });
    if (!response.ok) {
      // Avoid dumping a server response that could contain private transcripts.
      throw new Error(`AI server request failed (HTTP ${response.status})`);
    }
    return (await response.json()) as T;
  }

  capabilities(): Promise<AiCapabilities> {
    return this.request<AiCapabilities>("/v1/capabilities");
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
    return this.request<AiSession>(`/v1/sessions/${id}`);
  }
}
