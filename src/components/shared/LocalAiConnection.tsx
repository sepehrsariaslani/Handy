import { useEffect, useState } from "react";
import { DEFAULT_SERVER_URL } from "@/lib/ai/serverConfig";

type BackendState = "connecting" | "connected" | "unavailable";

/** Local-only connection indicator. Credentials never appear in the UI. */
export function LocalAiConnection({ apiKey }: { apiKey: string }) {
  const [state, setState] = useState<BackendState>("connecting");

  useEffect(() => {
    let active = true;
    const check = async () => {
      if (!apiKey) {
        if (active) setState("connecting");
        return;
      }
      try {
        const response = await fetch(DEFAULT_SERVER_URL + "/v1/capabilities", {
          headers: { Authorization: "Bearer " + apiKey },
        });
        if (active) {
          setState(response.ok ? "connected" : "unavailable");
        }
      } catch {
        if (active) setState("unavailable");
      }
    };

    void check();
    const interval = window.setInterval(() => void check(), 15_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [apiKey]);

  return (
    <p className="text-xs text-ink-3" role="status">
      {state === "connected"
        ? "Local AI connected"
        : state === "connecting"
          ? "Connecting to local AI..."
          : "Local AI unavailable — check the Handy Dev terminal"}
    </p>
  );
}
