import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Mic, Square, Copy, RadioTower } from "lucide-react";
import type { AiTone } from "@/lib/ai/contracts";
import {
  LiveStreamSession,
  loadLastLiveSession,
  type LiveResult,
} from "@/lib/ai/liveSession";
import { loadServerUrl, saveServerUrl } from "@/lib/ai/serverConfig";

const TONES: AiTone[] = ["verbatim", "professional", "formal", "casual", "friendly"];

export function LiveMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  const [apiKey, setApiKey] = useState("");
  const [language, setLanguage] = useState("fa");
  const [tone, setTone] = useState<AiTone>("professional");
  const [cleanup, setCleanup] = useState(true);
  const [consent, setConsent] = useState(false);
  const [savedSession, setSavedSession] = useState(loadLastLiveSession);
  const [phase, setPhase] = useState("idle");
  const [partial, setPartial] = useState("");
  const [results, setResults] = useState<Record<number, LiveResult>>({});
  const [busy, setBusy] = useState(false);
  const controller = useRef<LiveStreamSession | null>(null);

  useEffect(() => () => {
    const current = controller.current;
    if (current) void current.stop().finally(() => current.dispose());
  }, []);

  const active = ["recording", "reconnecting", "draining"].includes(phase);
  const transcript = Object.values(results)
    .sort((a, b) => a.sequence - b.sequence)
    .map((entry) => entry.segment.clean_text ?? entry.segment.raw_text)
    .filter(Boolean)
    .join(" ");

  const start = async () => {
    if (!consent || !apiKey.trim() || busy || active) return;
    setBusy(true);
    try {
      const url = saveServerUrl(serverUrl);
      setServerUrl(url);
      const next = new LiveStreamSession(
        url,
        apiKey,
        { language, tone, cleanup },
        {
          state: setPhase,
          result: (record) =>
            setResults((all) => ({ ...all, [record.sequence]: record })),
          partial: setPartial,
          error: (detail) =>
            toast.error(t("live.streamError"), { description: detail }),
        },
      );
      controller.current = next;
      const id = await next.start(savedSession ?? undefined);
      setSavedSession(id);
      setPhase("recording");
    } catch (error) {
      toast.error(t("live.startError"), {
        description: error instanceof Error ? error.message : String(error),
      });
      controller.current?.dispose();
      controller.current = null;
      setPhase("idle");
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      await controller.current?.stop();
      setPhase("draining");
    } catch (error) {
      toast.error(t("live.streamError"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };

  const newSession = () => {
    if (active) return;
    localStorage.removeItem("handy.live.sessionId.v1");
    setSavedSession(null);
    setResults({});
    setPartial("");
    setPhase("idle");
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(transcript);
      toast.success(t("live.copied"));
    } catch {
      toast.error(t("live.copyError"));
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto flex flex-col gap-5 pb-8">
      <div>
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <RadioTower size={22} /> {t("live.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-2">{t("live.description")}</p>
      </div>
      <section className="w-full rounded-xl border border-line p-4 flex flex-col gap-4">
        <label className="text-sm flex flex-col gap-1">
          {t("live.serverUrl")}
          <input type="url" value={serverUrl} disabled={active || busy}
            className="rounded-lg border border-line bg-transparent px-3 py-2"
            onChange={(event) => setServerUrl(event.target.value)} />
        </label>
        <label className="text-sm flex flex-col gap-1">
          {t("live.apiKey")}
          <input type="password" value={apiKey} disabled={active || busy}
            autoComplete="off" className="rounded-lg border border-line bg-transparent px-3 py-2"
            onChange={(event) => setApiKey(event.target.value)} />
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label className="text-sm flex flex-col gap-1">
            {t("live.language")}
            <input value={language} disabled={active || busy} maxLength={35}
              className="rounded-lg border border-line bg-transparent px-3 py-2"
              onChange={(event) => setLanguage(event.target.value)} />
          </label>
          <label className="text-sm flex flex-col gap-1">
            {t("live.tone")}
            <select value={tone} disabled={active || busy}
              className="rounded-lg border border-line bg-background px-3 py-2"
              onChange={(event) => setTone(event.target.value as AiTone)}>
              {TONES.map((item) => (
                <option key={item} value={item}>
                  {t("live.tones." + item)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="text-sm flex gap-2 items-start">
          <input type="checkbox" checked={cleanup} disabled={active || busy}
            onChange={(event) => setCleanup(event.target.checked)} />
          {t("live.cleanup")}
        </label>
        <label className="text-sm flex gap-2 items-start">
          <input type="checkbox" checked={consent} disabled={active || busy}
            onChange={(event) => setConsent(event.target.checked)} />
          {t("live.consent")}
        </label>
        {savedSession && !active && (
          <p className="text-sm text-ink-3">{t("live.resumeNotice")}</p>
        )}
        <div className="flex flex-wrap items-center gap-2">
          {!active ? (
            <button type="button" disabled={!consent || !apiKey.trim() || busy}
              className="bg-logo-primary text-white rounded-lg px-4 py-2 disabled:opacity-50 flex items-center gap-2"
              onClick={() => void start()}>
              <Mic size={16} /> {savedSession ? t("live.resume") : t("live.start")}
            </button>
          ) : (
            <button type="button" disabled={busy || phase === "draining"}
              className="bg-red-600 text-white rounded-lg px-4 py-2 disabled:opacity-50 flex items-center gap-2"
              onClick={() => void stop()}>
              <Square size={16} /> {t("live.stop")}
            </button>
          )}
          {!active && savedSession && (
            <button type="button" disabled={busy} onClick={newSession}
              className="rounded-lg border border-line px-3 py-2 text-sm">
              {t("live.newSession")}
            </button>
          )}
          <span className="text-sm text-ink-3">
            {t("live.status")}: {t("live.states." + phase, { defaultValue: phase })}
          </span>
        </div>
        <p className="text-xs text-ink-3">{t("live.offlineNotice")}</p>
      </section>
      <section className="w-full border border-line rounded-xl p-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold">{t("live.transcript")}</h2>
          <button type="button" disabled={!transcript} onClick={() => void copy()}
            className="border border-line px-3 py-2 rounded-lg disabled:opacity-50 text-sm flex items-center gap-2">
            <Copy size={15} /> {t("live.copy")}
          </button>
        </div>
        <div className="max-h-96 overflow-y-auto whitespace-pre-wrap select-text text-sm">
          {transcript}
          {partial && <span className="text-ink-3 italic"> {partial}</span>}
          {!transcript && !partial && <p className="text-ink-3">{t("live.empty")}</p>}
        </div>
        <p className="text-xs text-ink-3">{t("live.review")}</p>
      </section>
    </div>
  );
}
