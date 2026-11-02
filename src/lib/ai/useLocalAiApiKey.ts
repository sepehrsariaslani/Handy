import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { isLocalAiServer } from "./serverConfig";

/**
 * Retrieve the existing local token from the native app, not from web storage.
 * This is scoped to the exact 127.0.0.1:8765 endpoint and discarded when
 * switching to a remote/custom server.
 */
export function useLocalAiApiKey(serverUrl: string): string {
  const [local, setLocal] = useState<{ url: string; key: string } | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    if (!isLocalAiServer(serverUrl)) {
      setLocal(null);
      return;
    }
    void invoke<string | null>("get_local_ai_api_key", { serverUrl })
      .then((key) => {
        if (!cancelled) {
          setLocal(key ? { url: serverUrl, key } : null);
        }
      })
      .catch(() => {
        if (!cancelled) setLocal(null);
      });

    return () => {
      cancelled = true;
    };
  }, [serverUrl]);

  return isLocalAiServer(serverUrl) && local?.url === serverUrl
    ? local.key
    : "";
}
