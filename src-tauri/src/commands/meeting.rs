use crate::managers::audio::AudioRecordingManager;
use chrono::Utc;
use handy_recorder::{AudioChunk, Channels, EndReason, Recorder, RecorderConfig, Sink};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};

const SAMPLE_RATE: u32 = 16_000;
const FLUSH_EVERY_SAMPLES: u64 = SAMPLE_RATE as u64 * 5;

#[derive(Clone, Debug, Deserialize, Serialize, specta::Type)]
pub struct MeetingInfo {
    pub id: String,
    pub started_at: String,
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
    pub status: String,
    pub stage: Option<String>,
    pub error: Option<String>,
    pub transcript: Option<Vec<MeetingUtterance>>,
    pub insights: Option<MeetingInsights>,
}

struct ActiveMeeting {
    recorder: Recorder,
    info: MeetingInfo,
    directory: PathBuf,
}

#[derive(Default)]
pub struct MeetingState {
    active: Mutex<Option<ActiveMeeting>>,
}

impl MeetingState {
    pub fn is_active(&self) -> bool {
        self.active.lock().map(|meeting| meeting.is_some()).unwrap_or(false)
    }
}

/// Writes straight to disk: a 20-hour session never builds up a Vec<f32>.
/// WAV headers are periodically flushed to reduce loss on an unclean exit.
struct MeetingWavSink {
    writer: Option<hound::WavWriter<BufWriter<File>>>,
    samples_since_flush: u64,
    error: Option<String>,
}

impl MeetingWavSink {
    fn open(path: &Path) -> Result<Self, String> {
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
) -> Result<MeetingInfo, String> {
    let mut active = state.active.lock().map_err(|e| e.to_string())?;
    if active.is_some() {
        return Err("A meeting is already recording".to_owned());
    }
    if app.state::<Arc<AudioRecordingManager>>().is_recording() {
        return Err("Stop the current dictation before starting a meeting".to_owned());
    }
    let id = format!("{}-{}", Utc::now().format("%Y%m%dT%H%M%S%3f"), std::process::id());
    let directory = meeting_dir(&app)?.join(&id);
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let audio_path = directory.join("audio.wav");
    let mut info = MeetingInfo {
        id,
        started_at: Utc::now().to_rfc3339(),
        audio_path: audio_path.to_string_lossy().to_string(),
        bytes: 0,
        status: "recording".to_owned(),
        remote_job_id: None,
        remote_server_url: None,
    };
    let sink = MeetingWavSink::open(&audio_path)?;
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
            channels: settings.selected_channel.map_or(Channels::MixToMono, Channels::Only),
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
pub fn list_meetings(app: AppHandle, state: State<'_, MeetingState>) -> Result<Vec<MeetingInfo>, String> {
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
                let recording_this = state.active.lock().map_err(|e| e.to_string())?
                    .as_ref().is_some_and(|m| m.info.id == info.id);
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

fn server_endpoint(server_url: &str, suffix: &str) -> Result<reqwest::Url, String> {
    let base = reqwest::Url::parse(server_url).map_err(|e| e.to_string())?;
    let local = matches!(base.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if base.scheme() != "https" && !(base.scheme() == "http" && local) {
        return Err("Use HTTPS for remote servers; plain HTTP is allowed only on localhost".into());
    }
    if !base.username().is_empty() || base.password().is_some() {
        return Err("Credentials must not be embedded in the server URL".into());
    }
    base.join(suffix).map_err(|e| e.to_string())
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60 * 60))
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
    let form = reqwest::multipart::Form::new()
        .file("audio", &audio)
        .await
        .map_err(|e| e.to_string())?;
    let url = server_endpoint(&server_url, "v1/meetings")?;
    let mut request = client()?.post(url).multipart(form);
    if let Some(key) = api_key.filter(|s| !s.is_empty()) {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.map_err(|e| format!("Upload failed: {e}"))?;
    let status = response.status();
    let body = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("Server rejected the audio ({status}): {body}"));
    }
    let job = serde_json::from_str::<MeetingJob>(&body).map_err(|e| e.to_string())?;
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
