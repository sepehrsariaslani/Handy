/** Shared non-secret server preference; never persist an API token here. */
const STORAGE_KEY = "handy.ai.serverUrl.v1";
export const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

/**
 * Local-first development only. A future production server can be enabled
 * without touching Meeting/Notes/Memory/Tasks: set VITE_HANDY_AI_REMOTE=1
 * at build time and use the existing remote server URL/key controls.
 */
export const LOCAL_AI_TEST_MODE =
  import.meta.env.DEV && import.meta.env.VITE_HANDY_AI_REMOTE !== "1";

export function normalizeServerUrl(value: string): string {
  const url = new URL(value.trim());
  const isLoopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && isLoopback)) {
    throw new Error("Remote AI servers require HTTPS");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error(
      "Server URL must not contain credentials, query or fragment",
    );
  }
  if (url.pathname !== "/") {
    throw new Error("Enter the server root URL, without a path");
  }
  return url.origin;
}

/** The only origin allowed to use the existing local API token automatically. */
export function isLocalAiServer(value: string): boolean {
  try {
    const url = new URL(normalizeServerUrl(value));
    return (
      url.protocol === "http:" &&
      ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) &&
      url.port === "8765"
    );
  } catch {
    return false;
  }
}

export function loadServerUrl(): string {
  // Stale preferences from previous development sessions must not make a
  // local test app send its requests to an unrelated remote endpoint.
  if (LOCAL_AI_TEST_MODE) return DEFAULT_SERVER_URL;
  try {
    if (typeof window === "undefined") return DEFAULT_SERVER_URL;
    return normalizeServerUrl(
      window.localStorage.getItem(STORAGE_KEY) ?? DEFAULT_SERVER_URL,
    );
  } catch {
    return DEFAULT_SERVER_URL;
  }
}

export function saveServerUrl(value: string): string {
  const normalized = normalizeServerUrl(value);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, normalized);
  }
  return normalized;
}
