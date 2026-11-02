import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, CheckSquare, ChevronDown, Plus, RefreshCw, Save, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { HandyAiClient } from "@/lib/ai/client";
import type { AiTask, AiTaskInput, AiTaskStatus } from "@/lib/ai/contracts";
import { LOCAL_AI_TEST_MODE, loadServerUrl, saveServerUrl } from "@/lib/ai/serverConfig";
import { useLocalAiApiKey } from "@/lib/ai/useLocalAiApiKey";
import { LocalAiConnection } from "@/components/shared/LocalAiConnection";

const toError = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

type TaskGroup = {
  key: string;
  title: string;
  date: string | null;
  tasks: AiTask[];
};

export function TasksMode() {
  const { t } = useTranslation();
  const [serverUrl, setServerUrl] = useState(loadServerUrl);
  const [apiKey, setApiKey] = useState("");
  const localApiKey = useLocalAiApiKey(serverUrl);
  const authToken = apiKey.trim() || localApiKey;
  const [tasks, setTasks] = useState<AiTask[]>([]);
  const [filter, setFilter] = useState<AiTaskStatus | "all">("all");
  const [selected, setSelected] = useState<AiTask | null>(null);
  const [editing, setEditing] = useState(false);
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
      // Pagination is essential: a meeting can produce many tasks, and one
      // accordion must not silently lose tasks from older meetings.
      const all: AiTask[] = [];
      let total = 0;
      for (let offset = 0; offset < 5000; offset += 100) {
        const page = await client.listTasks(
          filter === "all" ? undefined : filter, offset, 100,
        );
        all.push(...page.items);
        total = page.total;
        if (all.length >= total || !page.items.length) break;
      }
      setTasks(all);
      setCount(total);
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
    }
  }, [client, filter, t]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const groups = useMemo(() => {
    const groupMap = new Map<string, TaskGroup>();
    for (const task of tasks) {
      const key = task.source_meeting_id || "manual";
      const group = groupMap.get(key) ?? {
        key,
        title: task.source_meeting_id
          ? task.source_meeting_title || t("meeting.title")
          : t("tasks.otherTasks", { defaultValue: "Other tasks" }),
        date: task.source_meeting_date || null,
        tasks: [],
      };
      group.tasks.push(task);
      groupMap.set(key, group);
    }
    return [...groupMap.values()].sort((a, b) => {
      if (a.key === "manual") return 1;
      if (b.key === "manual") return -1;
      return (b.date ?? "").localeCompare(a.date ?? "");
    });
  }, [tasks, t]);

  const choose = (task: AiTask) => {
    setSelected(task);
    setTitle(task.title);
    setAssignee(task.assignee ?? "");
    setDueDate(task.due_date ?? "");
    setStatus(task.status);
    setEditing(true);
  };

  const newTask = () => {
    setSelected(null);
    setTitle("");
    setAssignee("");
    setDueDate("");
    setStatus("todo");
    setEditing(true);
  };

  const save = async () => {
    if (!client || busy || !title.trim()) return;
    const input: AiTaskInput = {
      title: title.trim(),
      assignee: assignee.trim() || null,
      due_date: dueDate || null,
    };
    setBusy(true);
    try {
      if (selected) {
        await client.updateTask(selected, input, status);
      } else {
        await client.createTask(input);
      }
      await reload();
      setEditing(false);
      toast.success(t("tasks.saved"));
    } catch (error) {
      toast.error(t("tasks.error"), { description: toError(error) });
      await reload();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!client || !selected || busy || !window.confirm(t("tasks.deleteConfirm"))) return;
    setBusy(true);
    try {
      await client.deleteTask(selected.id);
      setSelected(null);
      setEditing(false);
      await reload();
      toast.success(t("tasks.deleted"));
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
        <p className="text-sm text-ink-3 mt-1">
          {t("tasks.autoHint", { defaultValue: "Action items from meetings appear here automatically." })}
        </p>
      </header>

      {LOCAL_AI_TEST_MODE && <LocalAiConnection apiKey={authToken} />}
      {!LOCAL_AI_TEST_MODE && (
        <details className="rounded-xl border border-line p-3 text-sm">
          <summary className="cursor-pointer">{t("tasks.serverUrl")}</summary>
          <div className="mt-3 flex flex-col gap-2">
            <input type="url" value={serverUrl}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              aria-label={t("tasks.serverUrl")}
              onChange={(event) => { setServerUrl(event.target.value); setApiKey(""); }}
              onBlur={() => {
                try { setServerUrl(saveServerUrl(serverUrl)); }
                catch (error) { toast.error(toError(error)); }
              }} />
            <input type="password" autoComplete="off" value={apiKey}
              placeholder={localApiKey ? "Local API key detected" : t("tasks.apiKey")}
              aria-label={t("tasks.apiKey")}
              className="border border-line bg-transparent rounded-lg px-3 py-2"
              onChange={(event) => setApiKey(event.target.value)} />
          </div>
        </details>
      )}

      {!editing ? (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs text-ink-3">{t("tasks.count", { count })}</span>
            <div className="flex flex-wrap gap-2">
              <select value={filter} aria-label={t("tasks.filter")}
                className="rounded-lg border border-line bg-background px-3 py-2 text-sm"
                onChange={(event) => setFilter(event.target.value as AiTaskStatus | "all")}>
                {(["all", "todo", "in_progress", "done", "cancelled"] as const).map((value) => (
                  <option key={value} value={value}>{t("tasks.status." + value)}</option>
                ))}
              </select>
              <button type="button" onClick={() => void reload()}
                aria-label={t("meeting.refresh")} className="rounded-lg border border-line p-2">
                <RefreshCw size={17} />
              </button>
              <button type="button" onClick={newTask}
                className="rounded-lg bg-logo-primary text-white px-3 py-2 text-sm inline-flex items-center gap-2">
                <Plus size={16} /> {t("tasks.new")}
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            {groups.map((group) => (
              <details key={group.key} defaultOpen={index === 0}
                className="rounded-xl border border-line overflow-hidden group">
                <summary className="cursor-pointer list-none flex items-center justify-between gap-3 p-3 hover:bg-wash/30">
                  <span className="min-w-0">
                    <span className="block text-sm font-medium truncate">{group.title}</span>
                    <span className="block text-xs text-ink-3 mt-1">
                      {group.date ? group.date + " · " : ""}
                      {group.tasks.length} {t("tasks.title")}
                    </span>
                  </span>
                  <ChevronDown size={18} className="shrink-0 text-ink-3 group-open:rotate-180 transition-transform" />
                </summary>
                <div className="flex flex-col gap-1 border-t border-line p-2">
                  {group.tasks.map((task) => (
                    <button key={task.id} type="button" onClick={() => choose(task)}
                      className="text-start rounded-lg px-3 py-2 hover:bg-wash/40">
                      <span className="block text-sm">{task.title}</span>
                      <span className="block text-xs text-ink-3 mt-1">
                        {t("tasks.status." + task.status)}
                        {task.assignee ? " · " + task.assignee : ""}
                        {task.due_date ? " · " + task.due_date : ""}
                      </span>
                    </button>
                  ))}
                </div>
              </details>
            ))}
            {!groups.length && (
              <p className="text-sm text-ink-3 py-8 text-center">{t("tasks.empty")}</p>
            )}
          </div>
        </section>
      ) : (
        <section className="rounded-xl border border-line p-4 flex flex-col gap-4">
          <button type="button" disabled={busy} onClick={() => setEditing(false)}
            className="inline-flex w-fit items-center gap-2 text-sm text-ink-2">
            <ArrowLeft size={16} /> {t("onboarding.shell.back")}
          </button>
          <label className="text-sm flex flex-col gap-1">
            {t("tasks.taskTitle")}
            <textarea rows={3} maxLength={500} value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="border border-line bg-transparent rounded-lg px-3 py-2" />
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="text-sm flex flex-col gap-1">
              {t("tasks.assignee")}
              <input maxLength={120} value={assignee}
                onChange={(event) => setAssignee(event.target.value)}
                className="border border-line bg-transparent rounded-lg px-3 py-2" />
            </label>
            <label className="text-sm flex flex-col gap-1">
              {t("tasks.dueDate")}
              <input type="date" value={dueDate}
                onChange={(event) => setDueDate(event.target.value)}
                className="border border-line bg-transparent rounded-lg px-3 py-2" />
            </label>
          </div>
          {selected && (
            <label className="text-sm flex flex-col gap-1">
              {t("tasks.state")}
              <select value={status} className="border border-line bg-background rounded-lg px-3 py-2"
                onChange={(event) => setStatus(event.target.value as AiTaskStatus)}>
                {(["todo", "in_progress", "done", "cancelled"] as const).map((value) => (
                  <option key={value} value={value}>{t("tasks.status." + value)}</option>
                ))}
              </select>
            </label>
          )}
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy || !client || !title.trim()}
              onClick={() => void save()}
              className="bg-logo-primary text-white rounded-lg px-4 py-2 text-sm flex gap-2 items-center disabled:opacity-50">
              <Save size={16} /> {t("tasks.save")}
            </button>
            {selected && (
              <button type="button" disabled={busy || !client} onClick={() => void remove()}
                className="border border-line rounded-lg px-3 py-2 text-sm flex gap-2 items-center disabled:opacity-50">
                <Trash2 size={16} /> {t("tasks.delete")}
              </button>
            )}
          </div>
          {selected?.source_meeting_id && (
            <details className="border-t border-line pt-3">
              <summary className="cursor-pointer text-sm font-medium">
                {t("tasks.evidence")} · {selected.source_meeting_title || t("meeting.title")}
              </summary>
              <div className="mt-3 flex flex-col gap-2">
                {selected.source_quotes.map((quote) => (
                  <blockquote key={quote.id} className="border-s-2 border-line ps-3 text-sm whitespace-pre-wrap">
                    <span className="text-xs text-ink-3">{quote.id}</span>
                    <p>{quote.text}</p>
                  </blockquote>
                ))}
                <p className="text-xs text-ink-3">{t("tasks.review")}</p>
              </div>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
