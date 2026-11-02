import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BookOpenCheck, Search, Sparkles, CalendarDays } from "lucide-react";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type {
  AiMemoryQuery,
  AiMemoryResponse,
  AiMemoryScope,
  AiMemorySource,
} from "@/lib/ai/contracts";
import { loadServerUrl, saveServerUrl } from "@/lib/ai/serverConfig";

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const timecode = (seconds: number) => {
  const total = Math.floor(seconds);
  return [
    Math.floor(total / 3600),
    Math.floor((total % 3600) / 60),
    total % 60,
  ]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
};

export function MemoryMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  // Bearer credentials must not be stored in persistent web storage.
  const [token, setToken] = useState("");
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<AiMemoryScope>("all");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AiMemoryResponse | null>(null);

  const client = useMemo(() => {
    if (!token.trim()) return null;
    try {
      return new HandyAiClient(serverUrl, token.trim());
    } catch {
      return null;
    }
  }, [serverUrl, token]);

  const submit = async (mode: "ask" | "search") => {
    if (busy || !client || !question.trim()) return;
    if (since && until && since > until) {
      toast.error(t("memory.dateError"));
      return;
    }
    const body: AiMemoryQuery = {
      question: question.trim(),
      scope,
      ...(since ? { since } : {}),
      ...(until ? { until } : {}),
    };
    setBusy(true);
    setResult(null);
    try {
      const answer =
        mode === "ask"
          ? await client.askMemory(body)
          : await client.searchMemory(body);
      setResult(answer);
    } catch (error) {
      toast.error(t("memory.requestError"), {
        description: errorText(error),
      });
    } finally {
      setBusy(false);
    }
  };

  const jumpToCitation = (sourceId: string) => {
    const item = result?.sources.find((source) => source.id === sourceId);
    if (!item) return;
    document.getElementById("memory-" + sourceId)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  };

  const selectedCitations = new Set(result?.citations ?? []);

  const displaySource = (source: AiMemorySource) => (
    <article
      key={source.id}
      id={"memory-" + source.id}
      className={
        "rounded-xl border p-3 flex flex-col gap-2 " +
        (selectedCitations.has(source.id)
          ? "border-logo-primary"
          : "border-line")
      }
    >
      <div className="flex justify-between flex-wrap items-center gap-2">
        <strong className="text-sm font-medium">{source.title}</strong>
        <span className="text-xs text-ink-3">
          {source.kind === "meeting"
            ? t("memory.meeting")
            : t("memory.note")}
          {" · "}
          {source.created_at.slice(0, 10)}
        </span>
      </div>
      <p className="text-sm whitespace-pre-wrap" dir="auto">
        {source.quote}
      </p>
      <p className="text-xs text-ink-3">
        {source.speaker && source.speaker !== "Unknown"
          ? source.speaker + " · "
          : ""}
        {source.start !== null
          ? timecode(source.start) + " · "
          : ""}
        {source.source_id}
        {selectedCitations.has(source.id) && (
          <span className="ms-2 font-semibold text-logo-primary">
            {t("memory.cited")}
          </span>
        )}
      </p>
    </article>
  );

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col gap-4 pb-8">
      <header>
        <h1 className="flex items-center gap-2 text-xl font-semibold">
          <BookOpenCheck size={22} /> {t("memory.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-1">{t("memory.intro")}</p>
      </header>

      <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-sm">
          {t("memory.serverUrl")}
          <input
            type="url"
            className="rounded-lg border border-line bg-transparent px-3 py-2"
            value={serverUrl}
            onChange={(event) => setServerUrl(event.target.value)}
            onBlur={() => {
              try {
                setServerUrl(saveServerUrl(serverUrl));
              } catch (error) {
                toast.error(errorText(error));
              }
            }}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          {t("memory.apiKey")}
          <input
            type="password"
            autoComplete="off"
            className="rounded-lg border border-line bg-transparent px-3 py-2"
            value={token}
            onChange={(event) => setToken(event.target.value)}
          />
        </label>
        <p className="text-xs text-ink-3">{t("memory.privacy")}</p>
      </section>

      <form
        className="rounded-xl border border-line p-4 flex flex-col gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          void submit("ask");
        }}
      >
        <label className="flex flex-col gap-1 text-sm">
          {t("memory.question")}
          <textarea
            className="w-full border border-line bg-transparent rounded-lg px-3 py-2 resize-y"
            rows={3}
            maxLength={500}
            placeholder={t("memory.placeholder")}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          {t("memory.scope")}
          <select
            className="rounded-lg border border-line bg-background px-3 py-2"
            value={scope}
            onChange={(event) => setScope(event.target.value as AiMemoryScope)}
          >
            <option value="all">{t("memory.scopes.all")}</option>
            <option value="meetings">{t("memory.scopes.meetings")}</option>
            <option value="notes">{t("memory.scopes.notes")}</option>
          </select>
        </label>
        <div className="flex flex-wrap gap-3">
          <label className="flex-1 min-w-36 flex flex-col gap-1 text-sm">
            <span className="flex gap-1 items-center">
              <CalendarDays size={14} /> {t("memory.since")}
            </span>
            <input
              type="date"
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={since}
              onChange={(event) => setSince(event.target.value)}
            />
          </label>
          <label className="flex-1 min-w-36 flex flex-col gap-1 text-sm">
            {t("memory.until")}
            <input
              type="date"
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={until}
              onChange={(event) => setUntil(event.target.value)}
            />
          </label>
        </div>
        <p className="text-xs text-ink-3">{t("memory.dateNotice")}</p>
        <div className="flex flex-wrap gap-2">
          <button
            type="submit"
            disabled={!client || busy || question.trim().length < 3}
            className="flex gap-2 items-center bg-logo-primary text-white rounded-lg px-4 py-2 text-sm disabled:opacity-50"
          >
            <Sparkles size={16} /> {busy ? t("memory.working") : t("memory.ask")}
          </button>
          <button
            type="button"
            onClick={() => void submit("search")}
            disabled={!client || busy || question.trim().length < 3}
            className="flex gap-2 items-center border border-line rounded-lg px-4 py-2 text-sm disabled:opacity-50"
          >
            <Search size={16} /> {t("memory.searchOnly")}
          </button>
        </div>
      </form>

      {result && (
        <section className="border border-line rounded-xl p-4 flex flex-col gap-4">
          <h2 className="font-semibold">{t("memory.result")}</h2>
          {(result.applied_since || result.applied_until) && (
            <p className="text-xs text-ink-3">
              {t("memory.appliedPeriod")}:{" "}
              {result.applied_since ?? "…"} — {result.applied_until ?? "…"}
            </p>
          )}
          {result.status === "answered" && result.answer ? (
            <p className="text-sm leading-7 whitespace-pre-wrap" dir="auto">
              {result.answer}
            </p>
          ) : (
            <p className="text-sm text-ink-3">
              {result.status === "not_found"
                ? t("memory.noAnswer")
                : t("memory.searchOnlyResult")}
            </p>
          )}
          {result.warning && (
            <p className="text-xs text-ink-3" role="status">
              {result.warning}
            </p>
          )}
          {result.citations.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <span className="text-sm font-medium">
                {t("memory.answerSources")}:
              </span>
              {result.citations.map((citation) => (
                <button
                  key={citation}
                  type="button"
                  className="text-xs underline text-logo-primary"
                  onClick={() => jumpToCitation(citation)}
                >
                  {citation}
                </button>
              ))}
            </div>
          )}
          <h3 className="text-sm font-semibold">
            {t("memory.retrievedSources", { count: result.sources.length })}
          </h3>
          <div className="flex flex-col gap-2">
            {result.sources.map(displaySource)}
          </div>
          <p className="text-xs text-ink-3">{t("memory.review")}</p>
        </section>
      )}
    </div>
  );
}
