use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::ModelManager;
use chrono::Utc;
use handy_recorder::{AudioChunk, Channels, EndReason, Recorder, RecorderConfig, Sink};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

const SAMPLE_RATE: u32 = 16_000;
const FLUSH_EVERY_SAMPLES: u64 = SAMPLE_RATE as u64 * 5;

fn default_meeting_kind() -> String {
    "meeting".into()
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingInfo {
    pub id: String,
    #[serde(default = "default_meeting_kind")]
    pub kind: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub asr_model_id: Option<String>,
    pub started_at: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub meeting_date: Option<String>,
    pub audio_path: String,
    pub bytes: u64,
    pub status: String,
    pub remote_job_id: Option<String>,
    pub remote_server_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingUtterance {
    pub id: String,
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingEvidence {
    pub text: String,
    pub source_ids: Vec<String>,
    pub assignee: Option<String>,
    pub due_date: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingInsights {
    pub summary: String,
    pub decisions: Vec<MeetingEvidence>,
    pub tasks: Vec<MeetingEvidence>,
    pub open_questions: Vec<MeetingEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingJob {
    pub id: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub asr_model_id: Option<String>,
    #[serde(default)]
    pub generated_note_id: Option<String>,
    #[serde(default)]
    pub clean_text: Option<String>,
    pub title: Option<String>,
    pub meeting_date: Option<String>,
    pub status: String,
    pub stage: Option<String>,
    pub error: Option<String>,
    pub transcript: Option<Vec<MeetingUtterance>>,
    pub insights: Option<MeetingInsights>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub processed_seconds: Option<f64>,
    #[serde(default)]
    pub transcript_count: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingTranscriptPage {
    pub items: Vec<MeetingUtterance>,
    pub offset: usize,
    pub total: usize,
}

#[derive(Debug, Deserialize)]
struct UploadState {
    id: Option<String>,
    offset: u64,
    total_bytes: u64,
    job_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UploadPartState {
    offset: u64,
}

struct ActiveMeeting {
    recorder: Recorder<MeetingWavSink>,
    info: MeetingInfo,
    directory: PathBuf,
}

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<ActiveMeeting>>,
}

impl MeetingState {
    pub fn is_active(&self) -> bool {
        self.active
            .lock()
            .map(|meeting| meeting.is_some())
            .unwrap_or(false)
    }
}

/// Writes straight to disk: a 20-hour session never builds up a Vec<f32>.
/// WAV headers are periodically flushed to reduce loss on an unclean exit.
struct MeetingWavSink {
    writer: Option<hound::WavWriter<BufWriter<File>>>,
    samples_since_flush: u64,
    error: Option<String>,
    app: AppHandle,
    level_energy: f32,
    level_samples: usize,
}

impl MeetingWavSink {
    fn open(path: &Path, app: AppHandle) -> Result<Self, String> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let writer = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
        Ok(Self {
            writer: Some(writer),
            samples_since_flush: 0,
            error: None,
            app,
            level_energy: 0.0,
            level_samples: 0,
        })
    }

    fn finish(mut self) -> Result<(), String> {
        let writer = self.writer.take();
        if let Some(writer) = writer {
            writer.finalize().map_err(|e| e.to_string())?;
        }
        self.error.map_or(Ok(()), Err)
    }
}

impl Sink for MeetingWavSink {
    fn process_chunk(&mut self, chunk: AudioChunk<'_>) {
        if self.error.is_some() {
            return;
        }
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        for &value in &chunk.samples[..chunk.valid_frames] {
            self.level_energy += value * value;
            self.level_samples += 1;
            if self.level_samples >= 2048 {
                let rms = (self.level_energy / self.level_samples as f32).sqrt();
                let _ = self.app.emit("meeting-audio-level", rms.min(1.0));
                self.level_energy = 0.0;
                self.level_samples = 0;
            }
            let sample = (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            if let Err(error) = writer.write_sample(sample) {
                self.error = Some(format!("Cannot write meeting audio: {error}"));
                break;
            }
            self.samples_since_flush += 1;
        }
        if self.error.is_none() && self.samples_since_flush >= FLUSH_EVERY_SAMPLES {
            if let Err(error) = writer.flush() {
                self.error = Some(format!("Cannot flush meeting audio: {error}"));
            }
            self.samples_since_flush = 0;
        }
    }
}

/// Pin a downloaded model for the locally hosted Meeting/Notes ASR worker.
///
/// The local server reads only from the configured registry directory. The
/// model is symlinked, never duplicated or downloaded a second time.
/// Without the shared dev-only directory, legacy remote servers retain their
/// existing configured ASR model; we do not claim to switch them.
fn register_local_asr_model(app: &AppHandle) -> Result<Option<String>, String> {
    let Some(directory) = std::env::var_os("HANDY_MEETING_MODEL_DIR") else {
        return Ok(None);
    };
    let selected = crate::settings::get_settings(app).selected_model;
    if selected.is_empty() {
        return Err("Select and download a transcription model before recording".into());
    }
    if selected.len() > 120
        || !selected
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        || selected.contains("..")
    {
        return Err("Selected model ID is not valid for Meeting/Notes".into());
    }
    let manager = app.state::<Arc<ModelManager>>();
    let path = manager
        .get_model_path(&selected)
        .map_err(|e| format!("Selected transcription model is not downloaded: {e}"))?;
    if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
        return Err(
            "Meeting/Notes currently require a downloaded GGUF model supported by transcribe.cpp"
                .into(),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let directory = PathBuf::from(directory);
        fs::create_dir_all(&directory)
            .map_err(|e| format!("Cannot prepare local ASR registry: {e}"))?;
        let link = directory.join(format!("{selected}.gguf"));
        match fs::read_link(&link) {
            Ok(existing) if existing == path => {}
            Ok(_) => {
                fs::remove_file(&link).map_err(|e| e.to_string())?;
                symlink(&path, &link).map_err(|e| e.to_string())?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                symlink(&path, &link).map_err(|e| e.to_string())?;
            }
            Err(error) => return Err(format!("Cannot register model: {error}")),
        }
        log::info!("Meeting/Notes ASR model selected: {}", selected);
        Ok(Some(selected))
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        let _ = path;
        Err("Automatic Meeting/Notes model selection is not configured on this platform".into())
    }
}

fn meeting_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(crate::portable::app_data_dir(app)
        .map_err(|e| format!("Cannot resolve app data directory: {e}"))?
        .join("meetings"))
}

fn write_info(path: &Path, info: &MeetingInfo) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(info).map_err(|e| e.to_string())?;
    let tmp = path.join("meeting.tmp");
    fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    fs::rename(tmp, path.join("meeting.json")).map_err(|e| e.to_string())
}

fn get_info(path: &Path) -> Result<MeetingInfo, String> {
    let bytes = fs::read(path.join("meeting.json")).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn start_meeting(
    app: AppHandle,
    state: State<'_, MeetingState>,
    title: Option<String>,
    meeting_date: Option<String>,
    kind: Option<String>,
    language: Option<String>,
) -> Result<MeetingInfo, String> {
    let kind = kind.unwrap_or_else(default_meeting_kind);
    if kind != "meeting" && kind != "note" {
        return Err("Invalid recording kind".into());
    }
    let title = title.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    if title.as_ref().is_some_and(|s| s.chars().count() > 180) {
        return Err("Meeting title is too long".into());
    }
    if let Some(ref value) = meeting_date {
        chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .map_err(|_| "Invalid meeting date (YYYY-MM-DD)")?;
    }
    let mut active = state.active.lock().map_err(|e| e.to_string())?;
    if active.is_some() {
        return Err("A meeting is already recording".to_owned());
    }
    if app.state::<Arc<AudioRecordingManager>>().is_recording() {
        return Err("Stop the current dictation before starting a meeting".to_owned());
    }
    let id = format!(
        "{}-{}",
        Utc::now().format("%Y%m%dT%H%M%S%3f"),
        std::process::id()
    );
    let asr_model_id = register_local_asr_model(&app)?;
    let directory = meeting_dir(&app)?.join(&id);
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let audio_path = directory.join("audio.wav");
    // Explicit language always wins. Otherwise reuse the language selected
    // for standard dictation, rather than re-detecting each recording.
    let requested_language = language
        .filter(|code| code != "auto" && !code.is_empty())
        .or_else(|| {
            let selected = crate::settings::get_settings(&app).selected_language;
            if selected == "auto" || selected.is_empty() {
                None
            } else {
                Some(selected)
            }
        });
    let language = requested_language.filter(|code| {
        code.len() <= 12 && code.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
    });
    let mut info = MeetingInfo {
        id,
        kind,
        language,
        asr_model_id,
        started_at: Utc::now().to_rfc3339(),
        title,
        meeting_date,
        audio_path: audio_path.to_string_lossy().to_string(),
        bytes: 0,
        status: "recording".to_owned(),
        remote_job_id: None,
        remote_server_url: None,
    };
    let sink = MeetingWavSink::open(&audio_path, app.clone())?;
    let settings = crate::settings::get_settings(&app);
    let requested_mic = if settings.clamshell_microphone.is_some()
        && crate::helpers::clamshell::is_clamshell().unwrap_or(false)
    {
        settings.clamshell_microphone.clone()
    } else {
        settings.selected_microphone.clone()
    };
    let device = match requested_mic {
        Some(name) => {
            let devices = handy_recorder::list_input_devices()
                .map_err(|e| format!("Cannot enumerate microphones: {e}"))?;
            Some(
                devices
                    .into_iter()
                    .find(|item| item.name == name)
                    .map(|item| item.id)
                    .ok_or_else(|| format!("Selected microphone is unavailable: {name}"))?,
            )
        }
        None => None,
    };
    let recorder = Recorder::open_with_failure_handler(
        RecorderConfig {
            device,
            channels: settings
                .selected_channel
                .map_or(Channels::MixToMono, Channels::Only),
            frames_per_chunk: Some(1024),
            ..RecorderConfig::speech()
        },
        |error| log::error!("Meeting microphone failed: {error}"),
    )
    .map_err(|e| format!("Cannot open microphone: {e}"))?;
    recorder
        .start(sink)
        .map_err(|e| format!("Cannot start meeting recorder: {}", e.error))?;
    info.status = "recording".to_owned();
    // Keep the recording live even if metadata cannot be written.
    *active = Some(ActiveMeeting {
        recorder,
        info: info.clone(),
        directory: directory.clone(),
    });
    if let Err(error) = write_info(&directory, &info) {
        log::warn!("Failed to write meeting metadata: {error}");
    }
    Ok(info)
}

#[tauri::command]
#[specta::specta]
pub fn stop_meeting(state: State<'_, MeetingState>) -> Result<MeetingInfo, String> {
    let mut active = state
        .active
        .lock()
        .map_err(|e| e.to_string())?
        .take()
        .ok_or_else(|| "No active meeting".to_owned())?;
    let stopped = active.recorder.stop().map_err(|e| e.to_string())?;
    let normal_end = matches!(&stopped.end_reason, EndReason::StopCalled);
    let dropped = stopped.dropped_frames;
    let sink_result = stopped.sink.finish();
    if let Err(error) = active.recorder.close() {
        log::warn!("Meeting microphone could not close: {error}");
    }
    let size = fs::metadata(&active.info.audio_path)
        .map_err(|e| e.to_string())?
        .len();
    active.info.bytes = size;
    log::info!(
        "Meeting capture stopped: id={} kind={} language={} bytes={} dropped_frames={} normal_end={}",
        active.info.id, active.info.kind,
        active.info.language.as_deref().unwrap_or("auto"),
        size, dropped, normal_end,
    );
    active.info.status = if normal_end && dropped == 0 && sink_result.is_ok() {
        "ready"
    } else {
        "incomplete"
    }
    .to_owned();
    write_info(&active.directory, &active.info)?;
    if let Err(error) = sink_result {
        return Err(format!("Meeting audio may be incomplete: {error}"));
    }
    if !normal_end || dropped > 0 {
        return Err(format!(
            "Meeting capture was interrupted (dropped frames: {dropped}); file is preserved"
        ));
    }
    Ok(active.info)
}

#[tauri::command]
#[specta::specta]
pub fn recover_meeting(
    app: AppHandle,
    state: State<'_, MeetingState>,
    id: String,
) -> Result<MeetingInfo, String> {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 80 {
        return Err("Invalid meeting ID".into());
    }
    if state
        .active
        .lock()
        .map_err(|e| e.to_string())?
        .as_ref()
        .is_some_and(|meeting| meeting.info.id == id)
    {
        return Err("Cannot recover an active recording".into());
    }
    let directory = meeting_dir(&app)?.join(&id);
    let mut info = get_info(&directory)?;
    if info.status != "recording" && info.status != "incomplete" {
        return Err("This recording does not need recovery".into());
    }
    let path = directory.join("audio.wav");
    let reader = hound::WavReader::open(&path)
        .map_err(|e| format!("WAV is not readable; original file is preserved: {e}"))?;
    if reader.duration() == 0 {
        return Err("The recording contains no recoverable audio".into());
    }
    info.bytes = fs::metadata(path).map_err(|e| e.to_string())?.len();
    info.status = "ready".into();
    write_info(&directory, &info)?;
    Ok(info)
}

#[tauri::command]
#[specta::specta]
pub fn list_meetings(
    app: AppHandle,
    state: State<'_, MeetingState>,
) -> Result<Vec<MeetingInfo>, String> {
    let root = meeting_dir(&app)?;
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut meetings = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if let Ok(mut info) = get_info(&path) {
            info.bytes = fs::metadata(&info.audio_path).map(|m| m.len()).unwrap_or(0);
            // A crashed process may have left a "recording" entry on disk.
            if info.status == "recording" {
                let recording_this = state
                    .active
                    .lock()
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .is_some_and(|m| m.info.id == info.id);
                if !recording_this {
                    info.status = "interrupted".into();
                }
            }
            meetings.push(info);
        }
    }
    meetings.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(meetings)
}

#[tauri::command]
#[specta::specta]
pub fn update_meeting_metadata(
    app: AppHandle,
    id: String,
    title: Option<String>,
    meeting_date: Option<String>,
) -> Result<MeetingInfo, String> {
    if id.is_empty() || id.len() > 80 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return Err("Invalid meeting ID".into());
    }
    let directory = meeting_dir(&app)?.join(&id);
    let mut info = get_info(&directory)?;
    // Never replace an explicit title/date. Only fill the blanks after Qwen.
    if info.title.is_none() {
        info.title = title
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty() && s.chars().count() <= 180);
    }
    if info.meeting_date.is_none() {
        info.meeting_date =
            meeting_date.filter(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok());
    }
    write_info(&directory, &info)?;
    Ok(info)
}

/// Return the existing local development API token only for the exact loopback
/// Handy endpoint. Never expose the token for a user-supplied remote URL.
#[tauri::command]
#[specta::specta]
pub fn get_local_ai_api_key(server_url: String) -> Option<String> {
    let url = reqwest::Url::parse(&server_url).ok()?;
    let loopback = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if url.scheme() != "http"
        || !loopback
        || url.port_or_known_default() != Some(8765)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".config")
            .join("handy")
            .join("local-api-key");
        let token = fs::read_to_string(path).ok()?;
        let trimmed = token.trim();
        if trimmed.len() >= 24 {
            return Some(trimmed.to_owned());
        }
    }
    None
}

fn server_endpoint(server_url: &str, suffix: &str) -> Result<reqwest::Url, String> {
    let base = reqwest::Url::parse(server_url).map_err(|e| e.to_string())?;
    let local = matches!(base.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if base.scheme() != "https" && !(base.scheme() == "http" && local) {
        return Err("Use HTTPS for remote servers; plain HTTP is allowed only on localhost".into());
    }
    if !base.username().is_empty() || base.password().is_some() {
        return Err("Credentials must not be embedded in the server URL".into());
    }
    if base.query().is_some() || base.fragment().is_some() || base.path() != "/" {
        return Err("Meeting server URL must not contain a path, query or fragment".into());
    }
    base.join(suffix).map_err(|e| e.to_string())
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60 * 60))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn submit_meeting(
    app: AppHandle,
    id: String,
    server_url: String,
    api_key: Option<String>,
) -> Result<MeetingJob, String> {
    // Only load files belonging to this app's meeting directory.
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') || id.len() > 80 {
        return Err("Invalid meeting ID".into());
    }
    let directory = meeting_dir(&app)?.join(&id);
    let mut info = get_info(&directory)?;
    if info.status != "ready" {
        return Err("Stop the recording successfully before uploading".into());
    }
    let audio = directory.join("audio.wav");
    let audio_size = fs::metadata(&audio).map_err(|e| e.to_string())?.len();
    let root = server_endpoint(&server_url, "v1/meetings/uploads")?;
    let http = client()?;
    let auth = api_key.as_deref().filter(|s| !s.is_empty());

    // The server uses the local meeting ID as its idempotency key. Repeating
    // this request after a network failure retrieves the durable byte offset.
    let mut request = http.post(root.clone()).json(&serde_json::json!({
        "meeting_id": id,
        "kind": info.kind,
        "language": info.language,
        "asr_model_id": info.asr_model_id,
        "total_bytes": audio_size,
        "title": info.title,
        "meeting_date": info.meeting_date,
    }));
    if let Some(key) = auth {
        request = request.bearer_auth(key);
    }
    let begin = request
        .send()
        .await
        .map_err(|e| format!("Cannot start meeting upload: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Server rejected meeting upload: {e}"))?
        .json::<UploadState>()
        .await
        .map_err(|e| e.to_string())?;

    if begin.total_bytes != audio_size {
        return Err("Server upload size does not match local recording".into());
    }

    let job: MeetingJob = if let Some(existing_job) = begin.job_id {
        get_meeting_job(server_url.clone(), api_key.clone(), existing_job).await?
    } else {
        let upload_id = begin.id.ok_or("Upload ID missing")?;
        if upload_id.len() != 32 || !upload_id.chars().all(|ch| ch.is_ascii_hexdigit()) {
            return Err("Invalid upload ID".into());
        }
        let part_url = server_endpoint(&server_url, &format!("v1/meetings/uploads/{upload_id}"))?;
        let mut offset = begin.offset;
        if offset > audio_size {
            return Err("Invalid upload offset from server".into());
        }
        let mut retry_count = 0usize;
        let mut source = File::open(&audio).map_err(|e| e.to_string())?;
        while offset < audio_size {
            source
                .seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            let take = (audio_size - offset).min(4 * 1024 * 1024) as usize;
            let mut bytes = vec![0; take];
            source.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            let mut request = http
                .put(part_url.clone())
                .header("X-Upload-Offset", offset.to_string())
                .body(bytes);
            if let Some(key) = auth {
                request = request.bearer_auth(key);
            }
            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    let part = response
                        .json::<UploadPartState>()
                        .await
                        .map_err(|e| e.to_string())?;
                    if part.offset <= offset || part.offset > audio_size {
                        return Err("Invalid upload progress from server".into());
                    }
                    offset = part.offset;
                    retry_count = 0;
                }
                Ok(response) if response.status() == reqwest::StatusCode::CONFLICT => {
                    retry_count += 1;
                }
                Ok(response) => {
                    return Err(format!(
                        "Meeting upload failed (HTTP {})",
                        response.status()
                    ));
                }
                Err(_) => {
                    retry_count += 1;
                }
            }
            if retry_count > 0 {
                if retry_count > 3 {
                    return Err("Meeting upload interrupted; retry the recording to resume".into());
                }
                let mut probe = http.get(part_url.clone());
                if let Some(key) = auth {
                    probe = probe.bearer_auth(key);
                }
                let state = probe
                    .send()
                    .await
                    .map_err(|_| "Cannot reconnect to meeting upload")?
                    .error_for_status()
                    .map_err(|_| "Cannot resume meeting upload")?
                    .json::<UploadState>()
                    .await
                    .map_err(|e| e.to_string())?;
                if state.offset > audio_size {
                    return Err("Server resumed past the end of the file".into());
                }
                offset = state.offset;
            }
        }
        let finish_url = server_endpoint(
            &server_url,
            &format!("v1/meetings/uploads/{upload_id}/finish"),
        )?;
        let mut request = http.post(finish_url);
        if let Some(key) = auth {
            request = request.bearer_auth(key);
        }
        request
            .send()
            .await
            .map_err(|e| format!("Cannot finish meeting upload: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Meeting upload finalize failed: {e}"))?
            .json::<MeetingJob>()
            .await
            .map_err(|e| e.to_string())?
    };
    info.remote_job_id = Some(job.id.clone());
    info.remote_server_url = Some(server_url);
    write_info(&directory, &info)?;
    Ok(job)
}

#[tauri::command]
#[specta::specta]
pub async fn get_meeting_job(
    server_url: String,
    api_key: Option<String>,
    job_id: String,
) -> Result<MeetingJob, String> {
    if job_id.len() != 32 || !job_id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid job ID".into());
    }
    let url = server_endpoint(&server_url, &format!("v1/meetings/{job_id}"))?;
    let mut request = client()?.get(url);
    if let Some(key) = api_key.filter(|s| !s.is_empty()) {
        request = request.bearer_auth(key);
    }
    request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn retry_meeting_job(
    server_url: String,
    api_key: Option<String>,
    job_id: String,
) -> Result<MeetingJob, String> {
    if job_id.len() != 32 || !job_id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid job ID".into());
    }
    let url = server_endpoint(&server_url, &format!("v1/meetings/{job_id}/retry"))?;
    let mut request = client()?.post(url);
    if let Some(key) = api_key.filter(|s| !s.is_empty()) {
        request = request.bearer_auth(key);
    }
    request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn get_meeting_transcript(
    server_url: String,
    api_key: Option<String>,
    job_id: String,
    offset: usize,
    limit: usize,
) -> Result<MeetingTranscriptPage, String> {
    if job_id.len() != 32 || !job_id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid job ID".into());
    }
    if limit == 0 || limit > 500 {
        return Err("Invalid page size".into());
    }
    let mut url = server_endpoint(&server_url, &format!("v1/meetings/{job_id}/transcript"))?;
    url.query_pairs_mut()
        .append_pair("offset", &offset.to_string())
        .append_pair("limit", &limit.to_string());
    let mut request = client()?.get(url);
    if let Some(key) = api_key.filter(|s| !s.is_empty()) {
        request = request.bearer_auth(key);
    }
    request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())
}
