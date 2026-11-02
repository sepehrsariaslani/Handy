import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  BookOpenText,
  Search,
  Plus,
  Save,
  Trash2,
  Sparkles,
  Link2,
  Mic,
  ArrowLeft,
} from "lucide-react";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type { AiNote, AiNoteSummary, AiNotesPage } from "@/lib/ai/contracts";
import {
  LOCAL_AI_TEST_MODE,
  loadServerUrl,
  saveServerUrl,
} from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";
import { LocalAiConnection } from "@/components/shared/LocalAiConnection";
import { RecordingPill } from "@/components/ui/RecordingPill";

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function NotesMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  // The bearer token is deliberately not written to localStorage.
  const [token, setToken] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = token.trim() || localApiKey;
  const [search, setSearch] = useState("");
  const [items, setItems] = useState<AiNoteSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [note, setNote] = useState<AiNote | null>(null);
  const [editorExpanded, setEditorExpanded] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [meetingId, setMeetingId] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [liveLevel, setLiveLevel] = useState(0);
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [recordingModel, setRecordingModel] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [speechLanguage, setSpeechLanguage] = useState("auto");
  const [pendingNotes, setPendingNotes] = useState<string[]>([]);
  const [liveStatus, setLiveStatus] = useState<
    "idle" | "recording" | "draining" | "saving" | "completed" | "failed"
  >("idle");
  const liveBusy =
    recording || liveStatus === "draining" || liveStatus === "saving";

  const client = useMemo(
    () => (authToken ? new HandyAiClient(serverUrl, authToken) : null),
    [serverUrl, authToken],
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

  // Durable WAV recording; Notes never use low-context live streaming.
  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;
    void listen<number>("meeting-audio-level", (event) => {
      if (active) setLiveLevel(event.payload);
    }).then((stop) => {
      if (active) unlisten = stop;
      else stop();
    });
    void invoke<Array<{ id: string; kind: string; status: string }>>(
      "list_meetings",
    )
      .then((rows) => {
        if (!active) return;
        setPendingNotes(
          rows
            .filter(
              (item) =>
                item.kind === "note" &&
                item.status === "ready" &&
                !(item as { remote_job_id?: string | null }).remote_job_id,
            )
            .map((item) => item.id),
        );
        const pending = rows.find(
          (item) => item.kind === "note" && item.status === "recording",
        );
        if (pending) {
          setRecordingId(pending.id);
          setRecording(true);
          setLiveStatus("recording");
          setEditorExpanded(true);
        }
      })
      .catch(() => undefined);
    return () => {
      active = false;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!jobId || !client || !authToken) return;
    let active = true;
    let finishing = false;
    const poll = async () => {
      if (finishing) return;
      try {
        const job = await invoke<{
          id: string;
          status: "queued" | "processing" | "completed" | "failed";
          error?: string | null;
          generated_note_id?: string | null;
        }>("get_meeting_job", { serverUrl, apiKey: authToken, jobId });
        if (!active) return;
        if (job.status === "failed") {
          finishing = true;
          setLiveStatus("failed");
          toast.error(job.error || t("notes.voiceFailed"));
        } else if (job.status === "completed") {
          finishing = true;
          if (!job.generated_note_id) {
            setLiveStatus("failed");
            toast.error("No speech was detected. The recording remains saved.");
            return;
          }
          const saved = await client.getRecordedNote(job.id);
          if (!active) return;
          setNote(saved);
          setTitle(saved.title);
          setContent(saved.content);
          setLiveStatus("completed");
          await loadList();
          try {
            const analyzed = await client.analyzeNote(saved.id);
            if (active) setNote(analyzed);
          } catch (error) {
            if (active)
              toast.error(t("notes.analysisFailed"), {
                description: message(error),
              });
          }
        }
      } catch (error) {
        if (active)
          toast.error("Recording sync", { description: message(error) });
      }
    };
    void poll();
    const timer = window.setInterval(() => {
      if (!finishing) void poll();
    }, 4000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [jobId, client, serverUrl, authToken, loadList, t]);

  const select = async (id: string) => {
    if (!client || busy || liveBusy) return;
    setBusy(true);
    try {
      const record = await client.getNote(id);
      setNote(record);
      setTitle(record.title);
      setContent(record.content);
      setEditorExpanded(true);
    } catch (error) {
      toast.error(t("notes.loadFailed"), { description: message(error) });
    } finally {
      setBusy(false);
    }
  };

  const newNote = () => {
    if (liveBusy) return;
    setNote(null);
    setTitle("");
    setContent("");
    setJobId(null);
    setLiveStatus("idle");
    setEditorExpanded(true);
  };

  const startLiveNote = async () => {
    if (!client || busy || liveBusy) return;
    setEditorExpanded(true);
    setLiveLevel(0);
    setJobId(null);
    setLiveStatus("recording");
    try {
      const info = await invoke<{ id: string; asr_model_id?: string | null }>("start_meeting", {
        kind: "note",
        title: title.trim() || null,
        meetingDate: null,
        language: speechLanguage === "auto" ? null : speechLanguage,
      });
      setRecordingId(info.id);
      setRecordingModel(info.asr_model_id ?? null);
      setRecording(true);
    } catch (error) {
      setLiveStatus("failed");
      toast.error("Cannot start note recording", {
        description: message(error),
      });
    }
  };

  const stopLiveNote = async () => {
    if (!recording || !recordingId) return;
    setRecording(false);
    setLiveLevel(0);
    setLiveStatus("saving");
    try {
      const info = await invoke<{ id: string }>("stop_meeting");
      setRecordingId(null);
      const job = await invoke<{ id: string }>("submit_meeting", {
        id: info.id,
        serverUrl,
        apiKey: authToken,
      });
      setJobId(job.id);
      setPendingNotes((current) => current.filter((id) => id !== info.id));
    } catch (error) {
      setLiveStatus("failed");
      toast.error("Note recording saved but processing did not start", {
        description: message(error),
      });
    }
  };

  const retrySavedRecording = async (id: string) => {
    if (!authToken || liveBusy || busy) return;
    setEditorExpanded(true);
    setLiveStatus("saving");
    try {
      const job = await invoke<{ id: string }>("submit_meeting", {
        id,
        serverUrl,
        apiKey: authToken,
      });
      setJobId(job.id);
      setPendingNotes((current) => current.filter((item) => item !== id));
    } catch (error) {
      setLiveStatus("failed");
      toast.error("Recording sync failed", { description: message(error) });
    }
  };

  const save = async () => {
    if (!client || !title.trim() || !content.trim() || busy || liveBusy) return;
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
    if (!client || !note || busy || liveBusy) return;
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
    if (
      !client ||
      !note ||
      busy ||
      liveBusy ||
      !window.confirm(t("notes.deleteConfirm"))
    )
      return;
    setBusy(true);
    try {
      await client.deleteNote(note.id);
      newNote();
      setEditorExpanded(false);
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
      setEditorExpanded(true);
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

      {LOCAL_AI_TEST_MODE && <LocalAiConnection apiKey={authToken} />}

      {!LOCAL_AI_TEST_MODE && (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            {t("notes.serverUrl")}
            <input
              type="url"
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={serverUrl}
              onChange={(event) => {
                setServerUrl(event.target.value);
                setToken("");
              }}
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
              placeholder={localApiKey ? "Local API key detected" : undefined}
              onChange={(event) => setToken(event.target.value)}
            />
          </label>
          <p className="text-xs text-ink-3">{t("notes.privacy")}</p>
        </section>
      )}

      {!editorExpanded && pendingNotes.length > 0 && (
        <section className="rounded-xl border border-line p-3 flex flex-col gap-2">
          <h2 className="text-sm font-medium">
            {t("notes.pendingRecordings", {
              defaultValue: "Saved recordings awaiting sync",
            })}
          </h2>
          {pendingNotes.map((id, index) => (
            <div
              key={id}
              className="flex items-center justify-between gap-2 text-sm"
            >
              <span>
                {t("notes.voiceTitle")} {index + 1}
              </span>
              <button
                type="button"
                disabled={!client || busy}
                onClick={() => void retrySavedRecording(id)}
                className="rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
              >
                {t("meeting.retrySync", {
                  defaultValue: "Resume transcription",
                })}
              </button>
            </div>
          ))}
        </section>
      )}
      {!editorExpanded && (
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
      )}

      <div
        className={
          editorExpanded
            ? "flex flex-col gap-4"
            : "grid grid-cols-1 md:grid-cols-[minmax(180px,1fr)_minmax(0,2fr)] gap-4"
        }
      >
        {!editorExpanded && (
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
              disabled={liveBusy}
              className="flex items-center justify-center gap-2 rounded-lg border border-line px-3 py-2 text-sm"
            >
              <Plus size={16} /> {t("notes.new")}
            </button>
            <div className="max-h-[420px] overflow-y-auto flex flex-col gap-1">
              {items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  disabled={busy || liveBusy}
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
        )}
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          {editorExpanded && (
            <button
              type="button"
              disabled={liveBusy}
              onClick={() => setEditorExpanded(false)}
              className="inline-flex w-fit items-center gap-2 text-sm text-ink-2 hover:text-text disabled:opacity-50"
            >
              <ArrowLeft size={16} />
              {t("onboarding.shell.back")}
            </button>
          )}
          <div className="flex flex-col gap-3 rounded-xl bg-wash/30 border border-line p-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Mic size={17} />
              {t("notes.voiceTitle")}
            </div>
            {!recording && (
              <label className="text-xs flex items-center gap-2 text-ink-3">
                {t("meeting.speechLanguage", {
                  defaultValue: "Recording language",
                })}
                <select
                  value={speechLanguage}
                  disabled={liveBusy}
                  onChange={(event) => setSpeechLanguage(event.target.value)}
                  className="border border-line bg-background rounded-lg px-2 py-1"
                >
                  <option value="auto">
                    {t("meeting.autoLanguage", {
                      defaultValue: "App setting / Auto detect",
                    })}
                  </option>
                  <option value="fa">
                    {t("meeting.persianLanguage", {
                      defaultValue: "فارسی (Persian)",
                    })}
                  </option>
                  <option value="en">
                    {t("meeting.englishLanguage", { defaultValue: "English" })}
                  </option>
                </select>
              </label>
            )}
            {recording && recordingModel && (
              <p className="text-xs text-ink-3">{t("meeting.asrModel", { defaultValue: "ASR model" })}: {recordingModel}</p>
            )}
            {recording ? (
              <RecordingPill
                level={liveLevel}
                onStop={() => void stopLiveNote()}
                label={t("notes.voiceTitle")}
              />
            ) : (
              <button
                type="button"
                disabled={!client || busy || liveBusy}
                onClick={() => void startLiveNote()}
                className="inline-flex w-fit items-center gap-2 rounded-full bg-logo-primary px-4 py-2 text-sm text-white disabled:opacity-50"
              >
                <Mic size={16} />
                {t("notes.startVoice")}
              </button>
            )}
            {liveStatus === "draining" && (
              <p className="text-sm text-ink-3" role="status">
                {t("notes.voiceDraining")}
              </p>
            )}
            {liveStatus === "saving" && (
              <p className="text-sm text-ink-3" role="status">
                {t("meeting.autoProcessing")}
              </p>
            )}
            {liveStatus === "completed" && (
              <p className="text-sm text-logo-primary" role="status">
                {t("notes.voiceReady")}
              </p>
            )}
            {liveStatus === "failed" && (
              <p className="text-sm text-red-500" role="alert">
                {t("notes.voiceFailed")}
              </p>
            )}
          </div>
          <input
            className="w-full border border-line bg-transparent rounded-lg px-3 py-2 font-semibold"
            placeholder={t("notes.noteTitle")}
            maxLength={180}
            value={title}
            onChange={(event) => setTitle(event.target.value)}
          />
          <textarea
            disabled={liveBusy}
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
                !client ||
                busy ||
                liveBusy ||
                !dirty ||
                !title.trim() ||
                !content.trim()
              }
              onClick={() => void save()}
              className="flex items-center gap-2 bg-logo-primary text-white rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <Save size={16} /> {t("notes.save")}
            </button>
            <button
              type="button"
              disabled={!client || busy || liveBusy || !note}
              onClick={() => void analyze()}
              className="flex gap-2 items-center rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
            >
              <Sparkles size={16} /> {t("notes.analyze")}
            </button>
            <button
              type="button"
              disabled={!client || busy || liveBusy || !note}
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
