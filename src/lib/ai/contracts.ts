/** Wire types matching server/handy_server/schemas.py (API v1). */
export const AI_MODES = ["dictation", "meeting", "note_taking"] as const;
export type AiMode = (typeof AI_MODES)[number];
export type AiTone =
  | "verbatim"
  | "professional"
  | "formal"
  | "casual"
  | "friendly";
export type AiSessionStatus =
  | "created"
  | "recording"
  | "processing"
  | "completed"
  | "failed";

export interface AiSessionCreate {
  mode: AiMode;
  title?: string | null;
  language?: string | null;
  tone?: AiTone;
}
export interface AiSession {
  id: string;
  mode: AiMode;
  status: AiSessionStatus;
  title: string | null;
  language: string | null;
  tone: AiTone;
  created_at: string;
}
export interface TranscriptSegment {
  id: string;
  start: number;
  end: number;
  speaker: string | null;
  raw_text: string;
  clean_text: string | null;
  final: boolean;
}
export interface SourceEvidence {
  text: string;
  source_ids: string[];
  assignee: string | null;
  due_date: string | null;
}
export interface MeetingAnalysis {
  summary: string;
  decisions: SourceEvidence[];
  tasks: SourceEvidence[];
  open_questions: SourceEvidence[];
}
export interface NotesAnalysis {
  title: string;
  sections: Record<string, string>[];
  key_points: SourceEvidence[];
  review_questions: string[];
}
export interface AiCapabilities {
  api_version: "v1";
  modes: AiMode[];
  processing: {
    asr: boolean;
    llm: boolean;
    speaker_diarization: boolean;
    streaming: boolean;
  };
  models: { asr: string; llm: string };
}
export interface AiModelStatus {
  asr_model: string;
  asr_ready: boolean;
  asr_device: string;
  llm_model: string;
  llm_ready: boolean;
}
export interface AiCleanupRequest {
  text: string;
  tone?: AiTone;
  language?: string | null;
  protected_terms?: string[];
}
export interface AiCleanupResponse {
  raw_text: string;
  clean_text: string;
  tone: AiTone;
  model: string;
  review_required: boolean;
  warnings: string[];
}
export interface AiTranscriptionResponse {
  raw_text: string;
  clean_text: string | null;
  language: string | null;
  model: string;
  segments: TranscriptSegment[];
  cleanup: AiCleanupResponse | null;
  elapsed_ms: number;
}

/** Phase-6 durable, single-tenant note with source provenance. */
export interface AiNoteSource {
  id: string;
  text: string;
}
export interface AiNote {
  id: string;
  title: string;
  content: string;
  source_meeting_id: string | null;
  analysis: NotesAnalysis | null;
  sources: AiNoteSource[];
  version: number;
  created_at: string;
  updated_at: string;
}
export interface AiNoteSummary {
  id: string;
  title: string;
  preview: string;
  source_meeting_id: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}
export interface AiNotesPage {
  items: AiNoteSummary[];
  total: number;
}


/** Phase-7 source-grounded retrieval across prior meetings and notes. */
export type AiMemoryScope = "all" | "notes" | "meetings";
export type AiMemoryStatus = "answered" | "search_only" | "not_found";
export interface AiMemoryQuery {
  question: string;
  scope?: AiMemoryScope;
  since?: string | null;
  until?: string | null;
}
export interface AiMemorySource {
  id: string;
  kind: "note" | "meeting";
  document_id: string;
  source_id: string;
  title: string;
  quote: string;
  created_at: string;
  speaker: string | null;
  start: number | null;
}
export interface AiMemoryResponse {
  question: string;
  scope: AiMemoryScope;
  status: AiMemoryStatus;
  answer: string | null;
  citations: string[];
  sources: AiMemorySource[];
  warning: string | null;
  applied_since: string | null;
  applied_until: string | null;
}
