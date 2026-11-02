import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { loadServerUrl, saveServerUrl } from "@/lib/ai/serverConfig";
import type { AiTone } from "@/lib/ai/contracts";

type Route = "local" | "server_with_fallback" | "server_only";
type GlobalConfig = {
  route: Route;
  server_url: string;
  cleanup: boolean;
  tone: AiTone;
};
type GlobalStatus = { config: GlobalConfig; has_token: boolean };

const DEFAULT_CONFIG: GlobalConfig = {
  route: "local",
  server_url: loadServerUrl(),
  cleanup: false,
  tone: "professional",
};
const ROUTES: Route[] = ["local", "server_with_fallback", "server_only"];
const TONES: AiTone[] = [
  "verbatim",
  "professional",
  "formal",
  "casual",
  "friendly",
];

export function GlobalDictationSettings() {
  const { t } = useTranslation();
  const [config, setConfig] = useState<GlobalConfig>(DEFAULT_CONFIG);
  const [token, setToken] = useState("");
  const [hasToken, setHasToken] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void invoke<GlobalStatus>("get_remote_dictation_status")
      .then((value) => {
        setConfig(value.config);
        setHasToken(value.has_token);
      })
      .catch(() => toast.error(t("live.global.loadError")));
  }, [t]);

  const save = async (clear = false) => {
    if (busy) return;
    setBusy(true);
    try {
      const url = saveServerUrl(config.server_url);
      const value = await invoke<GlobalStatus>("set_remote_dictation_config", {
        config: { ...config, server_url: url },
        token: clear ? "" : token || null,
      });
      setConfig(value.config);
      setToken("");
      setHasToken(value.has_token);
      toast.success(t("live.global.saved"));
    } catch (error) {
      toast.error(t("live.global.saveError"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="w-full rounded-xl border border-line p-4 flex flex-col gap-4">
      <div>
        <h2 className="font-semibold">{t("live.global.title")}</h2>
        <p className="text-xs text-ink-3 mt-1">
          {t("live.global.description")}
        </p>
      </div>
      <label className="text-sm flex flex-col gap-1">
        {t("live.global.route")}
        <select
          className="rounded-lg border border-line bg-background px-3 py-2"
          value={config.route}
          onChange={(event) =>
            setConfig((value) => ({
              ...value,
              route: event.target.value as Route,
            }))
          }
        >
          {ROUTES.map((route) => (
            <option key={route} value={route}>
              {t("live.global.routes." + route)}
            </option>
          ))}
        </select>
      </label>
      <label className="text-sm flex flex-col gap-1">
        {t("live.serverUrl")}
        <input
          type="url"
          className="rounded-lg border border-line bg-transparent px-3 py-2"
          value={config.server_url}
          onChange={(event) =>
            setConfig((value) => ({
              ...value,
              server_url: event.target.value,
            }))
          }
        />
      </label>
      <label className="text-sm flex flex-col gap-1">
        {t("live.global.token")}
        <input
          type="password"
          autoComplete="off"
          className="rounded-lg border border-line bg-transparent px-3 py-2"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          placeholder={
            hasToken ? t("live.global.tokenSet") : t("live.global.tokenMissing")
          }
        />
      </label>
      <label className="text-sm flex flex-col gap-1">
        {t("live.tone")}
        <select
          className="rounded-lg border border-line bg-background px-3 py-2"
          value={config.tone}
          onChange={(event) =>
            setConfig((value) => ({
              ...value,
              tone: event.target.value as AiTone,
            }))
          }
        >
          {TONES.map((style) => (
            <option value={style} key={style}>
              {t("live.tones." + style)}
            </option>
          ))}
        </select>
      </label>
      <label className="text-sm flex gap-2 items-start">
        <input
          type="checkbox"
          checked={config.cleanup}
          onChange={(event) =>
            setConfig((value) => ({ ...value, cleanup: event.target.checked }))
          }
        />
        {t("live.cleanup")}
      </label>
      <p className="text-xs text-ink-3">{t("live.global.safety")}</p>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={busy}
          className="border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
          onClick={() => void save()}
        >
          {t("live.global.save")}
        </button>
        {hasToken && (
          <button
            type="button"
            disabled={busy}
            className="border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            onClick={() => void save(true)}
          >
            {t("live.global.clearToken")}
          </button>
        )}
      </div>
    </section>
  );
}
