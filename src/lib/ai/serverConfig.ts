/** Shared non-secret server preference; never persist an API token here. */
const STORAGE_KEY = "handy.ai.serverUrl.v1";
export const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

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

export function loadServerUrl(): string {
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
