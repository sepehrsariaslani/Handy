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
