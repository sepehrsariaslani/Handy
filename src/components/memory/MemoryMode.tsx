import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  BookOpenCheck,
  Search,
  Sparkles,
  CalendarDays,
  Plus,
  Trash2,
  Database,
} from "lucide-react";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type {
  AiMemoryQuery,
  AiMemoryResponse,
  AiMemoryScope,
  AiMemorySource,
  AiConversation,
  AiConversationTurn,
  AiSemanticStatus,
} from "@/lib/ai/contracts";
import {
  LOCAL_AI_TEST_MODE,
  loadServerUrl,
  saveServerUrl,
} from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const timecode = (seconds: number) => {
  const total = Math.floor(seconds);
  return [Math.floor(total / 3600), Math.floor((total % 3600) / 60), total % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
};

export function MemoryMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  // Bearer credentials must not be stored in persistent web storage.
  const [token, setToken] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = token.trim() || localApiKey;
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<AiMemoryScope>("all");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AiMemoryResponse | null>(null);
  const [conversations, setConversations] = useState<AiConversation[]>([]);
  const [conversation, setConversation] = useState<AiConversation | null>(null);
  const [turns, setTurns] = useState<AiConversationTurn[]>([]);
  const [semanticStatus, setSemanticStatus] = useState<AiSemanticStatus | null>(
    null,
  );

  const client = useMemo(() => {
    if (!authToken) return null;
    try {
      return new HandyAiClient(serverUrl, authToken);
    } catch {
      return null;
    }
  }, [serverUrl, authToken]);

  useEffect(() => {
    let active = true;
    if (!client) {
      setConversations([]);
      setConversation(null);
      setTurns([]);
      setSemanticStatus(null);
      return;
    }
    void client
      .listConversations()
      .then((list) => {
        if (active) setConversations(list.items);
      })
      .catch(() => undefined);
    void client
      .getSemanticStatus()
      .then((status) => {
        if (active) setSemanticStatus(status);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [client]);

  const createConversation = async () => {
    if (!client || busy) return;
    setBusy(true);
    try {
      const fresh = await client.createConversation(
        t("memory.newConversation"),
      );
      setConversation(fresh);
      setTurns([]);
      setResult(null);
      const list = await client.listConversations();
      setConversations(list.items);
    } catch (error) {
      toast.error(t("memory.requestError"), { description: errorText(error) });
    } finally {
      setBusy(false);
    }
  };

  const openConversation = async (id: string) => {
    if (!client || busy) return;
    setBusy(true);
    try {
      const record = await client.getConversation(id);
      setConversation(record.conversation);
      setTurns(record.turns);
      setResult(null);
    } catch (error) {
      toast.error(t("memory.requestError"), { description: errorText(error) });
    } finally {
      setBusy(false);
    }
  };

  const deleteConversation = async () => {
    if (
      !client ||
      !conversation ||
      busy ||
      !window.confirm(t("memory.deleteConversationConfirm"))
    )
      return;
    setBusy(true);
    try {
      await client.deleteConversation(conversation.id);
      setConversation(null);
      setTurns([]);
      setResult(null);
      setConversations((current) =>
        current.filter((item) => item.id !== conversation.id),
      );
    } catch (error) {
      toast.error(t("memory.requestError"), { description: errorText(error) });
    } finally {
      setBusy(false);
    }
  };

  const indexSemantic = async () => {
    if (!client || !semanticStatus?.enabled || busy) return;
    setBusy(true);
    try {
      const result = await client.syncSemantic(100);
      setSemanticStatus(await client.getSemanticStatus());
      toast.success(t("memory.indexed", { count: result.indexed }));
    } catch (error) {
      toast.error(t("memory.requestError"), { description: errorText(error) });
    } finally {
      setBusy(false);
    }
  };

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
      if (mode === "ask" && conversation) {
        const response = await client.askInConversation(conversation, body);
        setConversation(response.conversation);
        setResult(response.result);
        const loaded = await client.getConversation(conversation.id);
        setTurns(loaded.turns);
        setConversations((current) =>
          current.map((item) =>
            item.id === loaded.conversation.id ? loaded.conversation : item,
          ),
        );
      } else {
        const answer =
          mode === "ask"
            ? await client.askMemory(body)
            : await client.searchMemory(body);
        setResult(answer);
      }
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
          {source.kind === "meeting" ? t("memory.meeting") : t("memory.note")}
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
        {source.start !== null ? timecode(source.start) + " · " : ""}
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

      {!LOCAL_AI_TEST_MODE && (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            {t("memory.serverUrl")}
            <input
              type="url"
              className="rounded-lg border border-line bg-transparent px-3 py-2"
              value={serverUrl}
              onChange={(event) => {
                setServerUrl(event.target.value);
                setToken("");
              }}
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
              placeholder={localApiKey ? "Local API key detected" : undefined}
              onChange={(event) => setToken(event.target.value)}
            />
          </label>
          <p className="text-xs text-ink-3">{t("memory.privacy")}</p>
        </section>
      )}

      <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
        <div className="flex flex-wrap gap-2 items-center justify-between">
          <h2 className="font-semibold">{t("memory.conversations")}</h2>
          <button
            type="button"
            disabled={!client || busy}
            onClick={() => void createConversation()}
            className="border border-line rounded-lg px-3 py-2 flex gap-2 items-center text-sm disabled:opacity-50"
          >
            <Plus size={16} /> {t("memory.newConversation")}
          </button>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <select
            value={conversation?.id ?? ""}
            disabled={!client || busy}
            className="flex-1 min-w-40 border border-line bg-background rounded-lg px-3 py-2 text-sm"
            onChange={(event) => {
              if (event.target.value) void openConversation(event.target.value);
              else {
                setConversation(null);
                setTurns([]);
                setResult(null);
              }
            }}
          >
            <option value="">{t("memory.oneOff")}</option>
            {conversations.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title} ({item.turn_count})
              </option>
            ))}
          </select>
          {conversation && (
            <button
              type="button"
              onClick={() => void deleteConversation()}
              disabled={busy}
              className="border border-line rounded-lg p-2 disabled:opacity-50"
              aria-label={t("memory.deleteConversation")}
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>
        {turns.length > 0 && (
          <div className="flex flex-col gap-3 max-h-72 overflow-y-auto">
            {turns.map((turn) => (
              <div key={turn.id} className="text-sm border-b border-line pb-2">
                <p className="font-medium whitespace-pre-wrap">
                  {turn.question}
                </p>
                <p className="text-ink-3 whitespace-pre-wrap mt-1">
                  {turn.answer || t("memory.noAnswer")}
                </p>
                {turn.citations.length > 0 && (
                  <p className="text-xs text-ink-3 mt-1">
                    {turn.citations.join(" · ")}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
      <section className="rounded-xl border border-line p-4 flex items-center gap-3 flex-wrap">
        <Database size={18} />
        <div className="flex-1 min-w-40">
          <h2 className="font-semibold text-sm">{t("memory.semantic")}</h2>
          <p className="text-xs text-ink-3">
            {semanticStatus?.enabled
              ? t("memory.semanticCount", { count: semanticStatus.indexed })
              : t("memory.semanticOff")}
          </p>
        </div>
        <button
          type="button"
          disabled={!client || busy || !semanticStatus?.enabled}
          onClick={() => void indexSemantic()}
          className="border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
        >
          {t("memory.syncSemantic")}
        </button>
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
            <Sparkles size={16} />{" "}
            {busy ? t("memory.working") : t("memory.ask")}
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
              {t("memory.appliedPeriod")}: {result.applied_since ?? "…"} —{" "}
              {result.applied_until ?? "…"}
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
