import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  BookOpenText,
  Search,
  Plus,
  Save,
  Trash2,
  Sparkles,
  Link2,
} from "lucide-react";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type { AiNote, AiNoteSummary, AiNotesPage } from "@/lib/ai/contracts";
import { loadServerUrl, saveServerUrl } from "@/lib/ai/serverConfig";

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function NotesMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  // The bearer token is deliberately not written to localStorage.
  const [token, setToken] = useState("");
  const [search, setSearch] = useState("");
  const [items, setItems] = useState<AiNoteSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [note, setNote] = useState<AiNote | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [meetingId, setMeetingId] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);

  const client = useMemo(
    () => (token.trim() ? new HandyAiClient(serverUrl, token.trim()) : null),
    [serverUrl, token],
  );

  const loadList = useCallback(async () => {
    if (!client) {
      setItems([]);
      setTotal(0);
      return;
    }
    setLoading(true);
    try {
      const result: AiNotesPage = await client.listNotes(search, page * 25, 25);
      setItems(result.items);
      setTotal(result.total);
    } catch (error) {
      toast.error(t("notes.loadFailed"), { description: message(error) });
    } finally {
      setLoading(false);
    }
  }, [client, search, page, t]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadList(), 280);
    return () => window.clearTimeout(timer);
  }, [loadList]);

  const select = async (id: string) => {
    if (!client || busy) return;
    setBusy(true);
    try {
      const record = await client.getNote(id);
      setNote(record);
      setTitle(record.title);
      setContent(record.content);
    } catch (error) {
      toast.error(t("notes.loadFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const newNote = () => {
    setNote(null);
    setTitle("");
    setContent("");
  };

  const save = async () => {
    if (!client || !title.trim() || !content.trim() || busy) return;
    setBusy(true);
    try {
      const saved = note
        ? await client.updateNote(note, title.trim(), content)
        : await client.createNote(title.trim(), content);
      setNote(saved);
      setTitle(saved.title);
      setContent(saved.content);
      toast.success(t("notes.saved"));
      await loadList();
    } catch (error) {
      toast.error(t("notes.saveFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const analyze = async () => {
    if (!client || !note || busy) return;
    if (title !== note.title || content !== note.content) {
      toast.error(t("notes.saveFirst"));
      return;
    }
    setBusy(true);
    try {
      const updated = await client.analyzeNote(note.id);
      setNote(updated);
      toast.success(t("notes.analyzed"));
    } catch (error) {
      toast.error(t("notes.analysisFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!client || !note || busy || !window.confirm(t("notes.deleteConfirm")))
      return;
    setBusy(true);
    try {
      await client.deleteNote(note.id);
      newNote();
      toast.success(t("notes.deleted"));
      await loadList();
    } catch (error) {
      toast.error(t("notes.deleteFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const importMeeting = async () => {
    if (!client || !/^[0-9a-f]{32}$/.test(meetingId) || busy) return;
    setBusy(true);
    try {
      const imported = await client.importMeetingNote(meetingId);
      setNote(imported);
      setTitle(imported.title);
      setContent(imported.content);
      setMeetingId("");
      toast.success(t("notes.imported"));
      await loadList();
    } catch (error) {
      toast.error(t("notes.importFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const dirty = note
    ? note.title !== title || note.content !== content
    : !!title || !!content;

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col gap-4 pb-8">
      <header>
        <h1 className="text-xl font-semibold flex gap-2 items-center">
          <BookOpenText size={23} /> {t("notes.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-1">{t("notes.intro")}</p>
      </header>

      <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-sm">
          {t("notes.serverUrl")}
          <input
            type="url"
            className="border border-line bg-transparent rounded-lg px-3 py-2"
            value={serverUrl}
            onChange={(event) => setServerUrl(event.target.value)}
            onBlur={() => {
              try {
                setServerUrl(saveServerUrl(serverUrl));
              } catch (error) {
                toast.error(message(error));
              }
            }}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          {t("notes.token")}
          <input
            type="password"
            autoComplete="off"
            className="border border-line bg-transparent rounded-lg px-3 py-2"
            value={token}
            onChange={(event) => setToken(event.target.value)}
          />
        </label>
        <p className="text-xs text-ink-3">{t("notes.privacy")}</p>
      </section>

      <section className="rounded-xl border border-line p-3 flex flex-wrap gap-2 items-end">
        <label className="flex-1 min-w-40 flex flex-col gap-1 text-sm">
          {t("notes.meetingId")}
          <input
            className="border border-line bg-transparent rounded-lg px-3 py-2"
            value={meetingId}
            maxLength={32}
            placeholder={t("notes.meetingPlaceholder")}
            onChange={(event) => setMeetingId(event.target.value.trim())}
          />
        </label>
        <button
          type="button"
          disabled={!client || busy || !/^[0-9a-f]{32}$/.test(meetingId)}
          onClick={() => void importMeeting()}
          className="rounded-lg border border-line px-3 py-2 flex gap-2 items-center disabled:opacity-50 text-sm"
        >
          <Link2 size={16} /> {t("notes.importMeeting")}
        </button>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(180px,1fr)_minmax(0,2fr)] gap-4">
        <section className="rounded-xl border border-line p-3 flex flex-col gap-3">
          <div className="flex gap-2 items-center">
            <Search size={17} className="text-ink-3" />
            <input
              className="min-w-0 flex-1 border border-line bg-transparent rounded-lg px-2 py-2 text-sm"
              placeholder={t("notes.search")}
              value={search}
              maxLength={160}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(0);
              }}
            />
          </div>
          <button
            type="button"
            onClick={newNote}
            className="flex items-center justify-center gap-2 rounded-lg border border-line px-3 py-2 text-sm"
          >
            <Plus size={16} /> {t("notes.new")}
          </button>
          <div className="max-h-[420px] overflow-y-auto flex flex-col gap-1">
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                disabled={busy}
                onClick={() => void select(item.id)}
                className={
                  "text-start rounded-lg p-3 hover:bg-line " +
                  (note?.id === item.id ? "bg-line" : "")
                }
              >
                <span className="block font-medium text-sm truncate">
                  {item.title}
                </span>
                <span className="block text-xs text-ink-3 mt-1 line-clamp-2">
                  {item.preview}
                </span>
              </button>
            ))}
            {!items.length && (
              <p className="text-xs text-ink-3 p-2">
                {loading ? t("notes.loading") : t("notes.empty")}
              </p>
            )}
          </div>
          <div className="flex items-center justify-between text-xs text-ink-3">
            <span>{t("notes.count", { count: total })}</span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                {t("notes.previous")}
              </button>
              <button
                type="button"
                disabled={(page + 1) * 25 >= total}
                onClick={() => setPage((p) => p + 1)}
              >
                {t("notes.next")}
              </button>
            </div>
          </div>
        </section>
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <input
            className="w-full border border-line bg-transparent rounded-lg px-3 py-2 font-semibold"
            placeholder={t("notes.noteTitle")}
            maxLength={180}
            value={title}
            onChange={(event) => setTitle(event.target.value)}
          />
          <textarea
            rows={10}
            className="w-full min-h-40 border border-line rounded-lg bg-transparent px-3 py-2 text-sm resize-y"
            placeholder={t("notes.content")}
            maxLength={12000}
            value={content}
            onChange={(event) => setContent(event.target.value)}
          />
          <div className="flex gap-2 flex-wrap">
            <button
              type="button"
              disabled={
                !client || busy || !dirty || !title.trim() || !content.trim()
              }
              onClick={() => void save()}
              className="flex items-center gap-2 bg-logo-primary text-white rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <Save size={16} /> {t("notes.save")}
            </button>
            <button
              type="button"
              disabled={!client || busy || !note}
              onClick={() => void analyze()}
              className="flex gap-2 items-center rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
            >
              <Sparkles size={16} /> {t("notes.analyze")}
            </button>
            <button
              type="button"
              disabled={!client || busy || !note}
              onClick={() => void remove()}
              className="flex gap-2 items-center rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
            >
              <Trash2 size={16} /> {t("notes.delete")}
            </button>
          </div>
          {note?.source_meeting_id && (
            <p className="text-xs text-ink-3 break-all">
              {t("notes.meetingSource")}: {note.source_meeting_id}
            </p>
          )}
          {note?.analysis && !dirty && (
            <div className="border-t border-line pt-3 flex flex-col gap-3">
              <h2 className="font-semibold">{t("notes.analysis")}</h2>
              {note.analysis.sections.map((section, index) => (
                <div key={index}>
                  <h3 className="text-sm font-medium">
                    {section.heading ?? ""}
                  </h3>
                  <p className="text-sm whitespace-pre-wrap">
                    {section.content ?? ""}
                  </p>
                </div>
              ))}
              <h3 className="text-sm font-medium">{t("notes.keyPoints")}</h3>
              {note.analysis.key_points.map((point, index) => (
                <div className="text-sm" key={index}>
                  <p>{point.text}</p>
                  <div className="text-xs text-ink-3 flex flex-wrap gap-2">
                    {point.source_ids.map((id) => {
                      const source = note.sources.find(
                        (entry) => entry.id === id,
                      );
                      return (
                        <span
                          key={id}
                          title={source?.text ?? ""}
                          className="underline"
                        >
                          {id}: {source?.text ?? ""}
                        </span>
                      );
                    })}
                  </div>
                </div>
              ))}
              <h3 className="text-sm font-medium">
                {t("notes.reviewQuestions")}
              </h3>
              {note.analysis.review_questions.map((question, i) => (
                <p className="text-sm" key={i}>
                  {question}
                </p>
              ))}
              <p className="text-xs text-ink-3">{t("notes.review")}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
