import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CheckSquare,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  FileText,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type { AiTask, AiTaskInput, AiTaskStatus } from "@/lib/ai/contracts";
import {
  LOCAL_AI_TEST_MODE,
  loadServerUrl,
  saveServerUrl,
} from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";

const toError = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function TasksMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  const [apiKey, setApiKey] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = apiKey.trim() || localApiKey;
  const [meetingId, setMeetingId] = useState("");
  const [tasks, setTasks] = useState<AiTask[]>([]);
  const [filter, setFilter] = useState<AiTaskStatus | "all">("all");
  const [selected, setSelected] = useState<AiTask | null>(null);
  const [title, setTitle] = useState("");
  const [assignee, setAssignee] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [status, setStatus] = useState<AiTaskStatus>("todo");
  const [busy, setBusy] = useState(false);
  const [count, setCount] = useState(0);
  const client = useMemo(() => {
    if (!authToken) return null;
    try {
      return new HandyAiClient(serverUrl, authToken);
    } catch {
      return null;
    }
  }, [authToken, serverUrl]);

  const reload = useCallback(async () => {
    if (!client) {
      setTasks([]);
      setCount(0);
      return;
    }
    try {
      const page = await client.listTasks(
        filter === "all" ? undefined : filter,
      );
      setTasks(page.items);
      setCount(page.total);
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
    }
  }, [client, filter, t]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const choose = (task: AiTask) => {
    setSelected(task);
    setTitle(task.title);
    setAssignee(task.assignee ?? "");
    setDueDate(task.due_date ?? "");
    setStatus(task.status);
  };

  const clear = () => {
    setSelected(null);
    setTitle("");
    setAssignee("");
    setDueDate("");
    setStatus("todo");
  };

  const save = async () => {
    if (!client || busy || !title.trim()) return;
    setBusy(true);
    const input: AiTaskInput = {
      title: title.trim(),
      assignee: assignee.trim() || null,
      due_date: dueDate || null,
    };
    try {
      const result = selected
        ? await client.updateTask(selected, input, status)
        : await client.createTask(input);
      choose(result);
      await reload();
      toast.success(t("tasks.saved"));
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
      // A stale optimistic version requires reloading latest task.
      await reload();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (
      !client ||
      !selected ||
      busy ||
      !window.confirm(t("tasks.deleteConfirm"))
    )
      return;
    setBusy(true);
    try {
      await client.deleteTask(selected.id);
      clear();
      await reload();
      toast.success(t("tasks.deleted"));
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
    } finally {
      setBusy(false);
    }
  };

  const importTasks = async () => {
    if (!client || busy || !/^[0-9a-f]{32}$/.test(meetingId)) return;
    setBusy(true);
    try {
      const result = await client.importMeetingTasks(meetingId);
      await reload();
      setMeetingId("");
      toast.success(t("tasks.imported", { count: result.created }));
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col gap-4 pb-8">
      <header>
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <CheckSquare size={22} /> {t("tasks.title")}
        </h1>
        <p className="text-sm text-ink-3 mt-1">{t("tasks.intro")}</p>
      </header>
      {!LOCAL_AI_TEST_MODE && (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            {t("tasks.serverUrl")}
            <input
              type="url"
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
                  toast.error(toError(error));
                }
              }}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            {t("tasks.apiKey")}
            <input
              type="password"
              autoComplete="off"
              value={apiKey}
              placeholder={localApiKey ? "Local API key detected" : undefined}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              onChange={(event) => setApiKey(event.target.value)}
            />
          </label>
          <p className="text-xs text-ink-3">{t("tasks.privacy")}</p>
        </section>
      )}

      <section className="rounded-xl border border-line p-3 flex gap-2 flex-wrap items-end">
        <label className="flex-1 min-w-44 text-sm flex flex-col gap-1">
          {t("tasks.meetingId")}
          <input
            className="border border-line bg-transparent rounded-lg px-3 py-2"
            value={meetingId}
            placeholder={t("tasks.meetingPlaceholder")}
            maxLength={32}
            onChange={(event) => setMeetingId(event.target.value.trim())}
          />
        </label>
        <button
          type="button"
          disabled={busy || !client || !/^[0-9a-f]{32}$/.test(meetingId)}
          onClick={() => void importTasks()}
          className="rounded-lg border border-line px-3 py-2 text-sm flex items-center gap-2 disabled:opacity-50"
        >
          <FileText size={16} /> {t("tasks.importMeeting")}
        </button>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(180px,1fr)_minmax(0,2fr)] gap-4">
        <section className="rounded-xl border border-line p-3 flex flex-col gap-3">
          <div className="flex flex-wrap gap-2 items-center justify-between">
            <label className="text-sm flex flex-col gap-1">
              {t("tasks.filter")}
              <select
                value={filter}
                className="rounded-lg border border-line bg-background px-2 py-2"
                onChange={(event) =>
                  setFilter(event.target.value as AiTaskStatus | "all")
                }
              >
                <option value="all">{t("tasks.status.all")}</option>
                <option value="todo">{t("tasks.status.todo")}</option>
                <option value="in_progress">
                  {t("tasks.status.in_progress")}
                </option>
                <option value="done">{t("tasks.status.done")}</option>
                <option value="cancelled">{t("tasks.status.cancelled")}</option>
              </select>
            </label>
            <button
              type="button"
              className="rounded-lg border border-line p-2"
              onClick={() => void reload()}
            >
              <RefreshCw size={16} />
            </button>
          </div>
          <button
            type="button"
            onClick={clear}
            className="flex justify-center items-center gap-2 border border-line rounded-lg px-3 py-2 text-sm"
          >
            <Plus size={16} /> {t("tasks.new")}
          </button>
          <p className="text-xs text-ink-3">{t("tasks.count", { count })}</p>
          <div className="flex flex-col gap-1 max-h-[440px] overflow-y-auto">
            {tasks.map((task) => (
              <button
                key={task.id}
                type="button"
                className={
                  "rounded-lg p-3 text-start hover:bg-line " +
                  (selected?.id === task.id ? "bg-line" : "")
                }
                onClick={() => choose(task)}
              >
                <span className="text-sm font-medium block">{task.title}</span>
                <span className="text-xs text-ink-3 block">
                  {t("tasks.status." + task.status)}
                  {task.assignee ? " · " + task.assignee : ""}
                  {task.due_date ? " · " + task.due_date : ""}
                </span>
              </button>
            ))}
            {!tasks.length && (
              <p className="text-sm text-ink-3">{t("tasks.empty")}</p>
            )}
          </div>
        </section>

        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <label className="text-sm flex flex-col gap-1">
            {t("tasks.taskTitle")}
            <textarea
              rows={3}
              maxLength={500}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>
          <label className="text-sm flex flex-col gap-1">
            {t("tasks.assignee")}
            <input
              maxLength={120}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={assignee}
              onChange={(event) => setAssignee(event.target.value)}
            />
          </label>
          <label className="text-sm flex flex-col gap-1">
            {t("tasks.dueDate")}
            <input
              type="date"
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              value={dueDate}
              onChange={(event) => setDueDate(event.target.value)}
            />
          </label>
          {selected && (
            <label className="text-sm flex flex-col gap-1">
              {t("tasks.state")}
              <select
                className="border border-line bg-background rounded-lg px-3 py-2"
                value={status}
                onChange={(event) =>
                  setStatus(event.target.value as AiTaskStatus)
                }
              >
                <option value="todo">{t("tasks.status.todo")}</option>
                <option value="in_progress">
                  {t("tasks.status.in_progress")}
                </option>
                <option value="done">{t("tasks.status.done")}</option>
                <option value="cancelled">{t("tasks.status.cancelled")}</option>
              </select>
            </label>
          )}
          <div className="flex gap-2 flex-wrap">
            <button
              type="button"
              disabled={busy || !client || !title.trim()}
              onClick={() => void save()}
              className="bg-logo-primary text-white rounded-lg px-3 py-2 text-sm flex gap-2 items-center disabled:opacity-50"
            >
              <Save size={16} /> {t("tasks.save")}
            </button>
            {selected && (
              <button
                type="button"
                disabled={busy || !client}
                onClick={() => void remove()}
                className="border border-line rounded-lg px-3 py-2 text-sm flex gap-2 items-center disabled:opacity-50"
              >
                <Trash2 size={16} /> {t("tasks.delete")}
              </button>
            )}
          </div>
          {selected?.source_meeting_id && (
            <div className="border-t border-line pt-3 flex flex-col gap-2">
              <h3 className="text-sm font-medium">{t("tasks.evidence")}</h3>
              <p className="text-xs text-ink-3 break-all">
                {t("tasks.fromMeeting")}: {selected.source_meeting_id}
              </p>
              {selected.source_quotes.map((quote) => (
                <blockquote
                  key={quote.id}
                  className="border-s-2 border-line ps-3 text-sm whitespace-pre-wrap"
                >
                  <span className="text-xs text-ink-3">{quote.id}</span>
                  <p>{quote.text}</p>
                </blockquote>
              ))}
              <p className="text-xs text-ink-3">{t("tasks.review")}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
