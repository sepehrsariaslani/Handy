import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import {
  Mic,
  UploadCloud,
  RefreshCw,
  Server,
  FileAudio,
  ClipboardList,
  BookOpenText,
  ListTodo,
} from "lucide-react";
import { toast } from "sonner";
import {
  LOCAL_AI_TEST_MODE,
  loadServerUrl,
  saveServerUrl,
} from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";
import { LocalAiConnection } from "@/components/shared/LocalAiConnection";
import { HandyAiClient } from "@/lib/ai/client";
import { RecordingPill } from "@/components/ui/RecordingPill";

type MeetingInfo = {
  id: string;
  started_at: string;
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
  status: "queued" | "processing" | "completed" | "failed";
  stage?: string;
  error?: string | null;
  transcript?: Utterance[];
  insights?: Insights;
  duration?: number;
  processed_seconds?: number;
  transcript_count?: number;
};

type TranscriptPage = {
  items: Utterance[];
  offset: number;
  total: number;
};

const readError = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const formatTime = (seconds: number) => {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return [Math.floor(minutes / 60), minutes % 60, secs]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
};

export const MeetingMode = () => {
  const { t } = useTranslation();
  const [meetings, setMeetings] = useState<MeetingInfo[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [level, setLevel] = useState(0);
  const [autoProcessing, setAutoProcessing] = useState(false);
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  // Token stays in memory; never store it in web storage or application logs.
  const [apiKey, setApiKey] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = apiKey.trim() || localApiKey;
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<MeetingJob | null>(null);
  const [visibleCount, setVisibleCount] = useState(150);
  const [loadingMore, setLoadingMore] = useState(false);
  const [sourcePreview, setSourcePreview] = useState<Utterance | null>(null);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
  const speakers = useMemo(
    () => [...new Set(job?.transcript?.map((entry) => entry.speaker) ?? [])],
    [job?.transcript],
  );

  const refresh = useCallback(async () => {
    const result = await invoke<MeetingInfo[]>("list_meetings");
    setMeetings(result);
    const active = result.find((item) => item.status === "recording");
    setCurrentId(active?.id ?? null);
  }, []);

  useEffect(() => {
    void refresh().catch((error) => toast.error(readError(error)));
  }, [refresh]);

  // The native meeting recorder sends actual RMS levels from the same audio
  // sink that writes the durable WAV. No second microphone is opened.
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
          apiKey: authToken || null,
          jobId,
        });
        if (alive)
          setJob((current) =>
            current?.id === result.id &&
            (current.transcript?.length ?? 0) > (result.transcript?.length ?? 0)
              ? { ...result, transcript: current.transcript }
              : result,
          );
      } catch (error) {
        if (alive) toast.error(readError(error));
      }
    };
    void load();
    const timer = window.setInterval(() => {
      if (job?.status !== "completed" && job?.status !== "failed") {
        void load();
      }
    }, 5000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [authToken, jobId, job?.status, serverUrl]);

  const start = async () => {
    if (!consent || busy) return;
    setBusy(true);
    try {
      const info = await invoke<MeetingInfo>("start_meeting");
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
      const meeting = await invoke<MeetingInfo>("stop_meeting");
      setCurrentId(null);
      setLevel(0);
      setConsent(false);
      await refresh();

      // Stop -> upload -> transcribe -> analyze with Qwen automatically.
      // A failed upload never deletes the safely saved local recording;
      // the Process button remains available for retry.
      if (!authToken) {
        toast.error(
          "Local AI is not connected. Recording was saved for retry.",
        );
        return;
      }
      setAutoProcessing(true);
      setJob(null);
      setJobId(null);
      setVisibleCount(150);
      setTranscriptOpen(true);
      setSpeakerNames({});
      const result = await invoke<MeetingJob>("submit_meeting", {
        id: meeting.id,
        serverUrl,
        apiKey: authToken,
      });
      setJobId(result.id);
      setJob(result);
    } catch (error) {
      toast.error(readError(error));
      setCurrentId(null);
      await refresh();
    } finally {
      setAutoProcessing(false);
      setBusy(false);
    }
  };

  const recover = async (meeting: MeetingInfo) => {
    if (busy || !window.confirm(t("meeting.recoverConfirm"))) return;
    setBusy(true);
    try {
      await invoke<MeetingInfo>("recover_meeting", { id: meeting.id });
      await refresh();
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const upload = async (meeting: MeetingInfo) => {
    if (busy) return;
    setBusy(true);
    setJobId(null);
    setJob(null);
    setVisibleCount(150);
    setTranscriptOpen(false);
    setSpeakerNames({});
    try {
      const result = await invoke<MeetingJob>("submit_meeting", {
        id: meeting.id,
        serverUrl,
        apiKey: authToken || null,
      });
      setJobId(result.id);
      setJob(result);
      toast.success(t("meeting.uploaded"));
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const retry = async () => {
    if (!jobId || busy) return;
    setBusy(true);
    try {
      const result = await invoke<MeetingJob>("retry_meeting_job", {
        serverUrl,
        apiKey: authToken || null,
        jobId,
      });
      setJob(result);
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setBusy(false);
    }
  };

  const importAsNote = async () => {
    if (!job || !job.insights || !authToken || busy) return;
    setBusy(true);
    try {
      const client = new HandyAiClient(serverUrl, authToken);
      await client.importMeetingNote(job.id);
      toast.success(t("notes.imported"));
    } catch (error) {
      toast.error(t("notes.importFailed"), { description: readError(error) });
    } finally {
      setBusy(false);
    }
  };

  const importAsTasks = async () => {
    if (
      !job ||
      job.status !== "completed" ||
      !job.insights ||
      !authToken ||
      busy
    )
      return;
    setBusy(true);
    try {
      const client = new HandyAiClient(serverUrl, authToken);
      const imported = await client.importMeetingTasks(job.id);
      toast.success(t("tasks.imported", { count: imported.created }));
    } catch (error) {
      toast.error(t("tasks.error"), { description: readError(error) });
    } finally {
      setBusy(false);
    }
  };

  const loadMore = async () => {
    if (!job || busy || loadingMore) return;
    const count = job.transcript?.length ?? 0;
    if (count >= (job.transcript_count ?? count)) {
      setVisibleCount((current) => Math.min(current + 150, count));
      return;
    }
    setLoadingMore(true);
    try {
      const page = await invoke<TranscriptPage>("get_meeting_transcript", {
        serverUrl,
        apiKey: authToken || null,
        jobId: job.id,
        offset: count,
        limit: 150,
      });
      setJob((current) => {
        if (!current || current.id !== job.id) return current;
        const merged = [...(current.transcript ?? []), ...page.items];
        const unique = new Map(merged.map((item) => [item.id, item]));
        return {
          ...current,
          transcript: [...unique.values()].sort((a, b) => a.start - b.start),
          transcript_count: page.total,
        };
      });
      setVisibleCount(count + page.items.length);
    } catch (error) {
      toast.error(readError(error));
    } finally {
      setLoadingMore(false);
    }
  };

  const jumpToSource = async (id: string) => {
    const index = job?.transcript?.findIndex((item) => item.id === id) ?? -1;
    setTranscriptOpen(true);
    if (index >= 0) {
      if (index >= visibleCount) setVisibleCount(index + 1);
      window.setTimeout(
        () =>
          document.getElementById("meeting-" + id)?.scrollIntoView({
            behavior: "smooth",
            block: "center",
          }),
        50,
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
      const found = page.items.find((entry) => entry.id === id);
      if (found) setSourcePreview(found);
    } catch (error) {
      toast.error(readError(error));
    }
  };

  const sources = (ids: string[]) => (
    <span className="inline-flex flex-wrap gap-1 ps-2">
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          className="text-xs text-logo-primary underline"
          onClick={() => void jumpToSource(id)}
        >
          {id}
        </button>
      ))}
    </span>
  );

  const evidence = (rows: Evidence[] | undefined, title: string) => (
    <section className="w-full">
      <h3 className="font-semibold mb-2">{title}</h3>
      {rows?.length ? (
        <ul className="flex flex-col gap-2">
          {rows.map((entry, index) => (
            <li
              key={index}
              className="rounded-lg border border-line p-3 text-sm"
            >
              <span>{entry.text}</span>
              {entry.assignee && (
                <span className="text-ink-3 ms-2">
                  · {t("meeting.assignee")}:{" "}
                  {speakerNames[entry.assignee] || entry.assignee}
                </span>
              )}
              {entry.due_date && (
                <span className="text-ink-3 ms-2">
                  · {t("meeting.dueDate")}: {entry.due_date}
                </span>
              )}
              {sources(entry.source_ids ?? [])}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-ink-3 text-sm">{t("meeting.none")}</p>
      )}
    </section>
  );

  return (
    <div className="w-full max-w-3xl flex flex-col gap-5 pb-8">
      <div className="w-full">
        <h1 className="text-xl font-semibold flex gap-2 items-center">
          <Mic size={22} />
          {t("meeting.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-2">{t("meeting.intro")}</p>
      </div>

      {LOCAL_AI_TEST_MODE && <LocalAiConnection apiKey={authToken} />}
      <div className="w-full border border-line rounded-xl p-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold">{t("meeting.recording")}</h2>
          <span className="text-xs text-ink-3">
            {currentId ? t("meeting.recordingNow") : t("meeting.stopped")}
          </span>
        </div>
        <label className="text-sm flex gap-2 items-start">
          <input
            type="checkbox"
            checked={consent}
            disabled={Boolean(currentId)}
            onChange={(event) => setConsent(event.target.checked)}
          />
          <span>{t("meeting.consent")}</span>
        </label>
        <div className="flex flex-wrap gap-2">
          {!currentId ? (
            <button
              type="button"
              disabled={!consent || busy}
              onClick={() => void start()}
              className="flex items-center gap-2 bg-logo-primary text-white rounded-lg px-4 py-2 text-sm disabled:opacity-50"
            >
              <Mic size={16} />
              {t("meeting.start")}
            </button>
          ) : (
            <RecordingPill
              level={level}
              disabled={busy}
              onStop={() => void stop()}
              label={t("meeting.recordingNow")}
            />
          )}
          <button
            type="button"
            onClick={() => void refresh()}
            className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm"
          >
            <RefreshCw size={16} />
            {t("meeting.refresh")}
          </button>
        </div>
        {autoProcessing && (
          <p className="text-sm text-logo-primary" role="status">
            {t("meeting.autoProcessing")}
          </p>
        )}
        <p className="text-xs text-ink-3">{t("meeting.storageNotice")}</p>
      </div>

      {!LOCAL_AI_TEST_MODE && (
        <div className="w-full border border-line rounded-xl p-4 flex flex-col gap-3">
          <h2 className="font-semibold flex items-center gap-2">
            <Server size={17} />
            {t("meeting.server")}
          </h2>
          <label className="flex flex-col gap-1 text-sm">
            {t("meeting.serverUrl")}
            <input
              className="rounded-lg border border-line bg-transparent px-3 py-2"
              type="url"
              value={serverUrl}
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
          </label>
          <label className="flex flex-col gap-1 text-sm">
            {t("meeting.apiKey")}
            <input
              className="rounded-lg border border-line bg-transparent px-3 py-2"
              type="password"
              autoComplete="off"
              value={apiKey}
              placeholder={localApiKey ? "Local API key detected" : undefined}
              onChange={(event) => setApiKey(event.target.value)}
            />
          </label>
          <p className="text-xs text-ink-3">{t("meeting.uploadNotice")}</p>
        </div>
      )}

      <section className="w-full flex flex-col gap-2">
        <h2 className="font-semibold flex items-center gap-2">
          <FileAudio size={17} />
          {t("meeting.saved")}
        </h2>
        {meetings.length === 0 && (
          <p className="text-ink-3 text-sm">{t("meeting.noRecordings")}</p>
        )}
        {meetings.map((meeting) => (
          <div
            key={meeting.id}
            className="border border-line rounded-lg p-3 flex flex-wrap justify-between gap-3 items-center"
          >
            <div className="min-w-0">
              <p className="text-sm font-medium">
                {new Date(meeting.started_at).toLocaleString()}
              </p>
              <p className="text-xs text-ink-3">
                {meeting.status} · {(meeting.bytes / 1024 / 1024).toFixed(1)} MB
              </p>
              <p className="text-xs text-ink-3 break-all">
                {meeting.audio_path}
              </p>
            </div>
            {["incomplete", "interrupted"].includes(meeting.status) && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void recover(meeting)}
                className="rounded-lg border border-line px-3 py-2 text-sm disabled:opacity-50"
              >
                {t("meeting.recover")}
              </button>
            )}
            <button
              type="button"
              disabled={busy || !authToken || meeting.status !== "ready"}
              onClick={() => {
                if (meeting.remote_job_id) {
                  setJob(null);
                  setVisibleCount(150);
                  setTranscriptOpen(true);
                  setSpeakerNames({});
                  setServerUrl(meeting.remote_server_url || serverUrl);
                  setJobId(meeting.remote_job_id);
                } else {
                  void upload(meeting);
                }
              }}
              className="flex items-center gap-2 border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <UploadCloud size={16} />
              {meeting.remote_job_id
                ? t("meeting.resume")
                : t("meeting.process")}
            </button>
          </div>
        ))}
      </section>

      {job && (
        <div className="w-full flex flex-col gap-4 border border-line rounded-xl p-4">
          <h2 className="font-semibold flex items-center gap-2">
            <ClipboardList size={18} />
            {t("meeting.results")}
          </h2>
          <p className="text-sm text-ink-3">
            {t("meeting.status")}: {job.status} · {job.stage ?? "-"}
            {job.duration && job.processed_seconds !== undefined
              ? ` · ${Math.floor((job.processed_seconds / job.duration) * 100)}%`
              : ""}
          </p>
          {Boolean(job.duration) && (
            <progress
              className="w-full h-2"
              aria-label={t("meeting.status")}
              max={job.duration}
              value={job.processed_seconds ?? 0}
            />
          )}
          {job.error && (
            <p role="alert" className="text-red-500 text-sm">
              {job.error}
            </p>
          )}
          {job.status === "failed" && (
            <button
              type="button"
              disabled={busy}
              onClick={() => void retry()}
              className="border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              {t("meeting.retry")}
            </button>
          )}
          {job.status === "completed" && job.insights && (
            <button
              type="button"
              disabled={busy || !authToken}
              onClick={() => void importAsNote()}
              className="flex items-center gap-2 border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <BookOpenText size={16} /> {t("notes.importMeeting")}
            </button>
          )}
          {job.status === "completed" && job.insights && (
            <button
              type="button"
              disabled={busy || !authToken}
              onClick={() => void importAsTasks()}
              className="flex items-center gap-2 border border-line rounded-lg px-3 py-2 text-sm disabled:opacity-50"
            >
              <ListTodo size={16} /> {t("tasks.importMeeting")}
            </button>
          )}
          {job.insights && (
            <>
              <section>
                <h3 className="font-semibold mb-2">{t("meeting.summary")}</h3>
                <p className="text-sm whitespace-pre-wrap">
                  {job.insights.summary}
                </p>
              </section>
              {evidence(job.insights.decisions, t("meeting.decisions"))}
              {evidence(job.insights.tasks, t("meeting.tasks"))}
              {evidence(job.insights.open_questions, t("meeting.questions"))}
            </>
          )}
          {sourcePreview && (
            <div className="border border-line p-3 rounded-lg text-sm">
              <span className="font-medium">
                {speakerNames[sourcePreview.speaker] || sourcePreview.speaker}
              </span>
              <span className="ms-2 text-xs text-ink-3">
                {formatTime(sourcePreview.start)} · {sourcePreview.id}
              </span>
              <p className="whitespace-pre-wrap mt-2">{sourcePreview.text}</p>
            </div>
          )}
          {speakers.length > 0 && (
            <section className="w-full flex flex-col gap-2">
              <h3 className="font-semibold">{t("meeting.speakerNames")}</h3>
              {speakers.map((speaker) => (
                <label
                  key={speaker}
                  className="flex items-center gap-3 text-sm"
                >
                  <span className="min-w-28">{speaker}</span>
                  <input
                    className="flex-1 rounded-lg border border-line bg-transparent px-3 py-2"
                    placeholder={t("meeting.speakerPlaceholder")}
                    value={speakerNames[speaker] ?? ""}
                    onChange={(event) =>
                      setSpeakerNames((current) => ({
                        ...current,
                        [speaker]: event.target.value,
                      }))
                    }
                  />
                </label>
              ))}
            </section>
          )}
          {job.transcript && (
            <details
              className="w-full"
              open={transcriptOpen || Boolean(job.error)}
              onToggle={(event) => setTranscriptOpen(event.currentTarget.open)}
            >
              <summary className="cursor-pointer font-semibold">
                {t("meeting.transcript")}
              </summary>
              <div className="max-h-96 overflow-y-auto mt-3 flex flex-col gap-2">
                {job.transcript.slice(0, visibleCount).map((item) => (
                  <p
                    key={item.id}
                    id={"meeting-" + item.id}
                    className="text-sm border-b border-line pb-2"
                  >
                    <span className="font-medium">
                      {speakerNames[item.speaker] || item.speaker}
                    </span>
                    <span className="text-xs text-ink-3 ms-2">
                      {formatTime(item.start)} · {item.id}
                    </span>
                    <span className="block whitespace-pre-wrap mt-1">
                      {item.text}
                    </span>
                  </p>
                ))}
              </div>
              {(job.transcript_count ?? job.transcript.length) >
                visibleCount && (
                <button
                  type="button"
                  className="mt-3 px-3 py-2 rounded-lg border border-line text-sm"
                  disabled={loadingMore}
                  onClick={() => void loadMore()}
                >
                  {t("meeting.showMore")}
                </button>
              )}
            </details>
          )}
          <p className="text-xs text-ink-3">{t("meeting.review")}</p>
        </div>
      )}
    </div>
  );
};
