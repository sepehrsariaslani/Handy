import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import {
  ArrowLeft,
  CalendarDays,
  ChevronRight,
  Mic,
  Plus,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import {
  LOCAL_AI_TEST_MODE,
  loadServerUrl,
  saveServerUrl,
} from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";
import { LocalAiConnection } from "@/components/shared/LocalAiConnection";
import { RecordingPill } from "@/components/ui/RecordingPill";
import { HandyAiClient } from "@/lib/ai/client";

type MeetingInfo = {
  id: string;
  kind?: "meeting" | "note";
  started_at: string;
  title: string | null;
  meeting_date: string | null;
  audio_path: string;
  bytes: number;
  status: "recording" | "ready" | "incomplete" | "interrupted";
  remote_job_id: string | null;
  remote_server_url: string | null;
};
type Utterance = {
  id: string;
  speaker: string;
  start: number;
  end: number;
  text: string;
};
type Evidence = {
  text: string;
  source_ids: string[];
  assignee?: string | null;
  due_date?: string | null;
};
type Insights = {
  summary: string;
  decisions: Evidence[];
  tasks: Evidence[];
  open_questions: Evidence[];
};
type MeetingJob = {
  id: string;
  clean_text?: string | null;
  title?: string | null;
  meeting_date?: string | null;
  status: "queued" | "processing" | "completed" | "failed";
  stage?: string;
  error?: string | null;
  transcript?: Utterance[];
  insights?: Insights | null;
  duration?: number;
  processed_seconds?: number;
  transcript_count?: number;
};
type TranscriptPage = { items: Utterance[]; offset: number; total: number };

const readError = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
const clock = (seconds: number) =>
  [
    Math.floor(seconds / 3600),
    Math.floor((seconds % 3600) / 60),
    Math.floor(seconds % 60),
  ]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");

export const MeetingMode = () => {
  const { t } = useTranslation();
  const [meetings, setMeetings] = useState<MeetingInfo[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [draftDate, setDraftDate] = useState("");
  const [speechLanguage, setSpeechLanguage] = useState("auto");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [level, setLevel] = useState(0);
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  const [apiKey, setApiKey] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = apiKey.trim() || localApiKey;
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<MeetingJob | null>(null);
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
  const [visibleCount, setVisibleCount] = useState(150);
  const [loadingMore, setLoadingMore] = useState(false);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const [sourcePreview, setSourcePreview] = useState<Utterance | null>(null);
  const [polishedParts, setPolishedParts] = useState<string[]>([]);
  const [polishedTotal, setPolishedTotal] = useState(0);
  const [polishedLoading, setPolishedLoading] = useState(false);
  const selected =
    meetings.find((meeting) => meeting.id === selectedId) ?? null;
  const recording = Boolean(currentId && currentId === selectedId);
  const speakers = useMemo(
    () => [...new Set(job?.transcript?.map((part) => part.speaker) ?? [])],
    [job?.transcript],
  );

  const refresh = useCallback(async () => {
    const result = await invoke<MeetingInfo[]>("list_meetings");
    setMeetings(result.filter((item) => item.kind !== "note"));
    const active = result
      .filter((item) => item.kind !== "note")
      .find((item) => item.status === "recording");
    setCurrentId(active?.id ?? null);
    if (active) {
      setSelectedId((old) => old ?? active.id);
      setExpanded(true);
    }
  }, []);

  useEffect(() => {
    void refresh().catch((error) => toast.error(readError(error)));
  }, [refresh]);

  // Reuse the recording sink's RMS signal; never open a second microphone.
  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;
    void listen<number>("meeting-audio-level", (event) => {
      if (active) setLevel(event.payload);
    }).then((stop) => {
      if (active) unlisten = stop;
      else stop();
    });
    return () => {
      active = false;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!jobId || !authToken) return;
    let alive = true;
    const load = async () => {
      try {
        const result = await invoke<MeetingJob>("get_meeting_job", {
          serverUrl,
          apiKey: authToken,
          jobId,
        });
        if (!alive) return;
        setJob((previous) =>
          previous?.id === result.id &&
          (previous.transcript?.length ?? 0) > (result.transcript?.length ?? 0)
            ? { ...result, transcript: previous.transcript }
            : result,
        );
        if (result.status === "completed" || result.status === "failed") {
          window.clearInterval(timer);
        }
      } catch (error) {
        if (alive) toast.error(readError(error));
      }
    };
    const timer = window.setInterval(() => void load(), 5000);
    void load();
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [jobId, authToken, serverUrl, job?.status]);

  // Fill missing local metadata only after the model has completed processing.
  useEffect(() => {
    if (
      !job ||
      job.status !== "completed" ||
      !selected ||
      selected.remote_job_id !== job.id
    )
      return;
    if (
      (selected.title || !job.title) &&
      (selected.meeting_date || !job.meeting_date)
    )
      return;
    let alive = true;
    void invoke<MeetingInfo>("update_meeting_metadata", {
      id: selected.id,
      title: job.title || null,
      meetingDate: job.meeting_date || null,
    })
      .then(() => {
        if (alive) void refresh();
      })
      .catch((error) => {
        if (alive) toast.error(readError(error));
      });
    return () => {
      alive = false;
    };
  }, [job, selected, refresh]);

  const resetResult = () => {
    setJob(null);
    setJobId(null);
    setSpeakerNames({});
    setVisibleCount(150);
    setTranscriptOpen(false);
    setSourcePreview(null);
    setPolishedParts([]);
    setPolishedTotal(0);
  };

  const upload = async (meeting: MeetingInfo) => {
    if (busy || uploading || !authToken || meeting.status !== "ready") return;
    setUploading(true);
    resetResult();
    try {
      const result = await invoke<MeetingJob>("submit_meeting", {
        id: meeting.id,
        serverUrl,
        apiKey: authToken,
      });
      setJob(result);
      setJobId(result.id);
      await refresh();
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setUploading(false);
    }
  };

  const newMeeting = () => {
    if (currentId || busy || uploading) return;
    setSelectedId(null);
    setDraftTitle("");
    setDraftDate("");
    setConsent(false);
    resetResult();
    setExpanded(true);
  };

  const openMeeting = (meeting: MeetingInfo) => {
    if (busy || uploading) return;
    setSelectedId(meeting.id);
    setDraftTitle(meeting.title || "");
    setDraftDate(meeting.meeting_date || "");
    resetResult();
    setExpanded(true);
    if (meeting.remote_job_id) {
      setServerUrl(meeting.remote_server_url || serverUrl);
      setJobId(meeting.remote_job_id);
    } else if (meeting.status === "ready" && authToken) {
      // Older offline recordings are resumed on open, not via Process.
      void upload(meeting);
    }
  };

  const start = async () => {
    if (!consent || busy || currentId) return;
    setBusy(true);
    try {
      const info = await invoke<MeetingInfo>("start_meeting", {
        title: draftTitle.trim() || null,
        meetingDate: draftDate || null,
        language: speechLanguage === "auto" ? null : speechLanguage,
      });
      setSelectedId(info.id);
      setCurrentId(info.id);
      await refresh();
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    if (busy || !currentId) return;
    setBusy(true);
    try {
      const info = await invoke<MeetingInfo>("stop_meeting");
      setCurrentId(null);
      setLevel(0);
      setConsent(false);
      await refresh();
      // A single Stop is sufficient: upload, Whisper, Qwen and task storage
      // all happen without a user-visible Process or Import step.
      if (!authToken) {
        toast.error(
          "Recording saved locally. Connect your AI server to sync it.",
        );
        return;
      }
      setUploading(true);
      resetResult();
      setTranscriptOpen(false);
      const result = await invoke<MeetingJob>("submit_meeting", {
        id: info.id,
        serverUrl,
        apiKey: authToken,
      });
      setJob(result);
      setJobId(result.id);
      await refresh();
    } catch (error) {
      toast.error(readError(error));
      await refresh().catch(() => undefined);
    } finally {
      setUploading(false);
      setBusy(false);
    }
  };

  const recover = async () => {
    if (!selected || busy || !window.confirm(t("meeting.recoverConfirm")))
      return;
    setBusy(true);
    try {
      await invoke("recover_meeting", { id: selected.id });
      await refresh();
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const retry = async () => {
    if (!jobId || busy || !authToken) return;
    setBusy(true);
    try {
      const result = await invoke<MeetingJob>("retry_meeting_job", {
        serverUrl,
        apiKey: authToken,
        jobId,
      });
      setJob(result);
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const loadMore = async () => {
    if (!job || loadingMore) return;
    const offset = job.transcript?.length ?? 0;
    if (offset >= (job.transcript_count ?? offset)) {
      setVisibleCount((current) => current + 150);
      return;
    }
    setLoadingMore(true);
    try {
      const page = await invoke<TranscriptPage>("get_meeting_transcript", {
        serverUrl,
        apiKey: authToken || null,
        jobId: job.id,
        offset,
        limit: 150,
      });
      setJob((current) => {
        if (!current || current.id !== job.id) return current;
        const combined = new Map(
          [...(current.transcript ?? []), ...page.items].map(
            (part) => [part.id, part] as const,
          ),
        );
        return {
          ...current,
          transcript: [...combined.values()].sort((a, b) => a.start - b.start),
          transcript_count: page.total,
        };
      });
      setVisibleCount(offset + page.items.length);
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setLoadingMore(false);
    }
  };

  const loadPolished = async () => {
    if (!job || !authToken || polishedLoading) return;
    setPolishedLoading(true);
    try {
      const client = new HandyAiClient(serverUrl, authToken);
      const result = await client.getCleanedTranscript(
        job.id,
        polishedParts.length,
        10,
      );
      setPolishedParts((current) => [...current, ...result.items]);
      setPolishedTotal(result.total);
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setPolishedLoading(false);
    }
  };

  const jumpToSource = async (id: string) => {
    setTranscriptOpen(true);
    const offset = job?.transcript?.findIndex((part) => part.id === id) ?? -1;
    if (offset >= 0) {
      setVisibleCount((current) => Math.max(current, offset + 1));
      window.setTimeout(
        () =>
          document
            .getElementById("meeting-" + id)
            ?.scrollIntoView({ behavior: "smooth", block: "center" }),
        60,
      );
      return;
    }
    if (!job) return;
    const ordinal = Number(id.match(/^u([0-9]{8})$/)?.[1]);
    if (!Number.isSafeInteger(ordinal) || ordinal <= 0) return;
    try {
      const page = await invoke<TranscriptPage>("get_meeting_transcript", {
        serverUrl,
        apiKey: authToken || null,
        jobId: job.id,
        offset: ordinal - 1,
        limit: 1,
      });
      setSourcePreview(page.items.find((part) => part.id === id) ?? null);
    } catch (error) {
      toast.error(readError(error));
    }
  };

  const evidence = (rows: Evidence[] | undefined, label: string) => (
    <details className="border-t border-line pt-3">
      <summary className="cursor-pointer text-sm font-medium">
        {label} · {rows?.length ?? 0}
      </summary>
      <div className="mt-3 flex flex-col gap-2">
        {rows?.map((item, index) => (
          <div key={index} className="rounded-lg bg-wash/20 px-3 py-2 text-sm">
            <p>{item.text}</p>
            {(item.assignee || item.due_date) && (
              <p className="text-xs text-ink-3 mt-1">
                {[
                  item.assignee &&
                    (speakerNames[item.assignee] || item.assignee),
                  item.due_date,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
            <div className="flex gap-2 flex-wrap mt-1">
              {item.source_ids?.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => void jumpToSource(id)}
                  className="text-xs text-logo-primary underline"
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        ))}
        {!rows?.length && (
          <p className="text-sm text-ink-3">{t("meeting.none")}</p>
        )}
      </div>
    </details>
  );

  const displayTitle =
    job?.title ||
    selected?.title ||
    draftTitle ||
    t("meeting.untitled", { defaultValue: "New meeting" });
  const displayDate =
    job?.meeting_date ||
    selected?.meeting_date ||
    (selected?.started_at ? selected.started_at.slice(0, 10) : null);

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col gap-4 pb-8">
      <header>
        <h1 className="text-xl font-semibold flex gap-2 items-center">
          <Mic size={22} /> {t("meeting.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-1">
          {t("meeting.autoHint", {
            defaultValue:
              "Record once. Transcript, summary and tasks are organized automatically.",
          })}
        </p>
      </header>
      {LOCAL_AI_TEST_MODE && <LocalAiConnection apiKey={authToken} />}
      {!LOCAL_AI_TEST_MODE && (
        <details className="rounded-xl border border-line p-3 text-sm">
          <summary className="cursor-pointer">{t("meeting.server")}</summary>
          <div className="mt-3 flex flex-col gap-2">
            <input
              type="url"
              aria-label={t("meeting.serverUrl")}
              value={serverUrl}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              onChange={(event) => {
                setServerUrl(event.target.value);
                setApiKey("");
              }}
              onBlur={() => {
                try {
                  setServerUrl(saveServerUrl(serverUrl));
                } catch (error) {
                  toast.error(readError(error));
                }
              }}
            />
            <input
              type="password"
              autoComplete="off"
              value={apiKey}
              aria-label={t("meeting.apiKey")}
              placeholder={
                localApiKey ? "Local API key detected" : t("meeting.apiKey")
              }
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              onChange={(event) => setApiKey(event.target.value)}
            />
          </div>
        </details>
      )}

      {!expanded ? (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-medium">{t("meeting.saved")}</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => void refresh()}
                aria-label={t("meeting.refresh")}
                className="border border-line rounded-lg p-2"
              >
                <RefreshCw size={16} />
              </button>
              <button
                type="button"
                disabled={Boolean(currentId)}
                onClick={newMeeting}
                className="bg-logo-primary text-white rounded-lg px-3 py-2 text-sm inline-flex items-center gap-2 disabled:opacity-50"
              >
                <Plus size={16} />{" "}
                {t("meeting.new", { defaultValue: "New meeting" })}
              </button>
            </div>
          </div>
          <div className="flex flex-col gap-1">
            {meetings.map((meeting) => (
              <button
                key={meeting.id}
                type="button"
                onClick={() => openMeeting(meeting)}
                className="rounded-lg text-start px-3 py-3 flex items-center justify-between gap-3 hover:bg-wash/30"
              >
                <span className="min-w-0">
                  <span className="block text-sm font-medium truncate">
                    {meeting.title ||
                      t("meeting.untitled", { defaultValue: "Meeting" })}
                  </span>
                  <span className="block text-xs text-ink-3 mt-1">
                    {meeting.meeting_date ||
                      new Date(meeting.started_at).toLocaleDateString()}
                    {" · "}
                    {meeting.status}
                  </span>
                </span>
                <ChevronRight size={17} className="text-ink-3 shrink-0" />
              </button>
            ))}
            {!meetings.length && (
              <p className="py-8 text-center text-sm text-ink-3">
                {t("meeting.noRecordings")}
              </p>
            )}
          </div>
        </section>
      ) : (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-4">
          <button
            type="button"
            disabled={recording || busy || uploading}
            onClick={() => {
              setExpanded(false);
              resetResult();
            }}
            className="inline-flex w-fit items-center gap-2 text-sm text-ink-2 disabled:opacity-50"
          >
            <ArrowLeft size={16} /> {t("onboarding.shell.back")}
          </button>
          {!selected ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="text-sm flex flex-col gap-1">
                {t("meeting.meetingName", {
                  defaultValue: "Meeting title (optional)",
                })}
                <input
                  value={draftTitle}
                  maxLength={180}
                  placeholder={t("meeting.autoName", {
                    defaultValue: "AI will suggest a title",
                  })}
                  onChange={(event) => setDraftTitle(event.target.value)}
                  className="border border-line bg-transparent rounded-lg px-3 py-2"
                />
              </label>
              <label className="text-sm flex flex-col gap-1">
                <span className="flex items-center gap-1">
                  <CalendarDays size={15} />
                  {t("meeting.meetingDate", {
                    defaultValue: "Date (optional)",
                  })}
                </span>
                <input
                  type="date"
                  value={draftDate}
                  onChange={(event) => setDraftDate(event.target.value)}
                  className="border border-line bg-transparent rounded-lg px-3 py-2"
                />
              </label>
              <label className="text-sm flex flex-col gap-1">
                {t("meeting.speechLanguage", { defaultValue: "Recording language" })}
                <select
                  value={speechLanguage}
                  onChange={(event) => setSpeechLanguage(event.target.value)}
                  className="border border-line bg-background rounded-lg px-3 py-2"
                >
                  <option value="auto">{t("meeting.autoLanguage", { defaultValue: "App setting / Auto detect" })}</option>
                  <option value="fa">{t("meeting.persianLanguage", { defaultValue: "فارسی (Persian)" })}</option>
                  <option value="en">{t("meeting.englishLanguage", { defaultValue: "English" })}</option>
                </select>
              </label>
            </div>
          ) : (
            <div>
              <h2 className="text-lg font-semibold">{displayTitle}</h2>
              <p className="text-xs text-ink-3 mt-1">
                {displayDate} · {selected.status}
              </p>
            </div>
          )}
          {(!selected || recording) && (
            <label className="text-sm flex gap-2 items-start">
              <input
                type="checkbox"
                checked={consent}
                disabled={recording}
                onChange={(event) => setConsent(event.target.checked)}
              />
              <span>{t("meeting.consent")}</span>
            </label>
          )}
          <div className="rounded-xl bg-wash/20 border border-line p-3 flex flex-col items-start gap-3">
            {recording ? (
              <RecordingPill
                level={level}
                disabled={busy}
                onStop={() => void stop()}
                label={t("meeting.recordingNow")}
              />
            ) : !selected ? (
              <button
                type="button"
                disabled={!consent || busy}
                onClick={() => void start()}
                className="rounded-full bg-logo-primary text-white px-4 py-2 text-sm inline-flex gap-2 items-center disabled:opacity-50"
              >
                <Mic size={16} /> {t("meeting.start")}
              </button>
            ) : (
              <p className="text-xs text-ink-3">
                {uploading
                  ? t("meeting.autoProcessing")
                  : t("meeting.storageNotice")}
              </p>
            )}
            {recording && (
              <p className="text-xs text-ink-3">{t("meeting.storageNotice")}</p>
            )}
          </div>

          {selected &&
            ["incomplete", "interrupted"].includes(selected.status) && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void recover()}
                className="w-fit rounded-lg border border-line px-3 py-2 text-sm"
              >
                {t("meeting.recover")}
              </button>
            )}
          {selected?.status === "ready" &&
            !selected.remote_job_id &&
            !uploading && (
              <button
                type="button"
                disabled={busy || !authToken}
                onClick={() => void upload(selected)}
                className="w-fit rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
              >
                {t("meeting.retrySync", {
                  defaultValue: "Resume automatic sync",
                })}
              </button>
            )}
          {(uploading || (job && job.status !== "completed")) && (
            <div role="status" className="flex flex-col gap-2">
              <p className="text-sm text-ink-3">
                {uploading
                  ? t("meeting.autoProcessing")
                  : `${t("meeting.status")}: ${job?.stage || job?.status}`}
              </p>
              {job?.duration && (
                <progress
                  className="w-full h-2"
                  max={job.duration}
                  value={job.processed_seconds ?? 0}
                />
              )}
              {job?.error && (
                <p className="text-red-500 text-sm">{job.error}</p>
              )}
              {job?.status === "failed" && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void retry()}
                  className="rounded-lg border border-line px-3 py-2 text-sm w-fit"
                >
                  {t("meeting.retry")}
                </button>
              )}
            </div>
          )}
          {job?.status === "completed" && job.clean_text && (
            <section className="flex flex-col gap-3">
              <h3 className="text-sm font-semibold">
                {t("meeting.polishedTranscript", {
                  defaultValue: "Clean transcript",
                })}
              </h3>
              <p
                dir="auto"
                className="text-sm whitespace-pre-wrap leading-relaxed"
              >
                {(polishedParts.length ? polishedParts : [job.clean_text]).join(
                  "\n\n",
                )}
              </p>
              {(polishedParts.length === 0 ||
                polishedParts.length < polishedTotal) && (
                <button
                  type="button"
                  disabled={polishedLoading}
                  onClick={() => void loadPolished()}
                  className="w-fit text-sm border border-line rounded-lg px-3 py-2 disabled:opacity-50"
                >
                  {t("meeting.showMore")}
                </button>
              )}
            </section>
          )}
          {job?.insights && (
            <div className="flex flex-col gap-4">
              <section>
                <h3 className="text-sm font-semibold mb-2">
                  {t("meeting.summary")}
                </h3>
                <p className="text-sm whitespace-pre-wrap">
                  {job.insights.summary}
                </p>
              </section>
              {evidence(job.insights.tasks, t("meeting.tasks"))}
              {evidence(job.insights.decisions, t("meeting.decisions"))}
              {evidence(job.insights.open_questions, t("meeting.questions"))}
              <p className="text-xs text-ink-3">
                {t("meeting.tasksSaved", {
                  defaultValue:
                    "Action items are saved automatically in Tasks. Review suggested owners and deadlines.",
                })}
              </p>
            </div>
          )}
          {job && (job.transcript_count || job.transcript?.length) ? (
            <details
              open={transcriptOpen}
              onToggle={(event) => setTranscriptOpen(event.currentTarget.open)}
              className="border-t border-line pt-3"
            >
              <summary className="cursor-pointer text-sm font-medium">
                {t("meeting.transcript")}
              </summary>
              {speakers.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {speakers.map((speaker) => (
                    <label
                      key={speaker}
                      className="text-xs flex items-center gap-2"
                    >
                      {speaker}
                      <input
                        className="rounded-lg border border-line bg-transparent px-2 py-1"
                        aria-label={speaker}
                        placeholder={t("meeting.speakerPlaceholder")}
                        value={speakerNames[speaker] || ""}
                        onChange={(event) =>
                          setSpeakerNames((current) => ({
                            ...current,
                            [speaker]: event.target.value,
                          }))
                        }
                      />
                    </label>
                  ))}
                </div>
              )}
              <div className="max-h-96 overflow-y-auto mt-3 flex flex-col gap-2">
                {job.transcript?.slice(0, visibleCount).map((part) => (
                  <div
                    key={part.id}
                    id={"meeting-" + part.id}
                    className="border-b border-line pb-2"
                  >
                    <p className="text-xs text-ink-3">
                      {speakerNames[part.speaker] || part.speaker}
                      {" · "}
                      {clock(part.start)} · {part.id}
                    </p>
                    <p className="text-sm whitespace-pre-wrap mt-1">
                      {part.text}
                    </p>
                  </div>
                ))}
                {sourcePreview && (
                  <div className="rounded-lg bg-wash/20 p-3 text-sm">
                    {sourcePreview.id} · {sourcePreview.text}
                  </div>
                )}
              </div>
              {(job.transcript_count ?? job.transcript?.length ?? 0) >
                visibleCount && (
                <button
                  type="button"
                  disabled={loadingMore}
                  onClick={() => void loadMore()}
                  className="mt-3 rounded-lg border border-line px-3 py-2 text-sm"
                >
                  {t("meeting.showMore")}
                </button>
              )}
            </details>
          ) : null}
        </section>
      )}
    </div>
  );
};
