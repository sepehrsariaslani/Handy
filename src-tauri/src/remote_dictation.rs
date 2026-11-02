//! Opt-in server dictation for global shortcuts; bearer token is RAM-only.
use crate::managers::transcription::Transcription;
use crate::settings::get_settings;
use reqwest::{multipart, Url};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::io::Cursor;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

const STORE_FILE: &str = "remote_dictation.json";
const MAX_WAV_BYTES: usize = 25 * 1024 * 1024;
static API_TOKEN: OnceLock<Mutex<Option<String>>> = OnceLock::new();

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DictationRoute {
    #[default]
    Local,
    ServerWithFallback,
    ServerOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct RemoteDictationConfig {
    pub route: DictationRoute,
    pub server_url: String,
    pub cleanup: bool,
    pub tone: String,
}

impl Default for RemoteDictationConfig {
    fn default() -> Self {
        Self {
            route: DictationRoute::Local,
            server_url: "http://127.0.0.1:8765".into(),
            cleanup: false,
            tone: "professional".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct RemoteDictationStatus {
    pub config: RemoteDictationConfig,
    pub has_token: bool,
}

fn token_store() -> &'static Mutex<Option<String>> {
    API_TOKEN.get_or_init(|| Mutex::new(None))
}

pub fn validate_url(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|_| "Invalid server URL".to_owned())?;
    let host = url.host_str().unwrap_or_default();
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "[::1]");
    if url.scheme() != "https" && !(loopback && url.scheme() == "http") {
        return Err("Remote dictation requires HTTPS outside localhost".into());
    }
    if !url.username().is_empty() || url.password().is_some() ||
        url.query().is_some() || url.fragment().is_some() ||
        url.path() != "/" {
        return Err("Server URL must be the origin without credentials, path, query or fragment".into());
    }
    Ok(url)
}

pub fn current_config(app: &AppHandle) -> RemoteDictationConfig {
    let Ok(store) = app.store(crate::portable::store_path(STORE_FILE)) else {
        return RemoteDictationConfig::default();
    };
    store.get("config").and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

#[tauri::command]
#[specta::specta]
pub fn get_remote_dictation_status(app: AppHandle) -> RemoteDictationStatus {
    RemoteDictationStatus {
        config: current_config(&app),
        has_token: token_store().lock().is_ok_and(|token| token.is_some()),
    }
}

/// Persist only non-secret preferences. None retains the RAM-only token;
/// an empty token string explicitly clears it. Token never returns to UI.
#[tauri::command]
#[specta::specta]
pub fn set_remote_dictation_config(
    app: AppHandle,
    config: RemoteDictationConfig,
    token: Option<String>,
) -> Result<RemoteDictationStatus, String> {
    validate_url(&config.server_url)?;
    if !matches!(config.tone.as_str(), "verbatim" | "professional" | "formal" | "casual" | "friendly") {
        return Err("Invalid writing tone".into());
    }
    if let Some(ref secret) = token {
        if !secret.is_empty() && secret.len() < 24 {
            return Err("Token must be at least 24 characters".into());
        }
    }
    let store = app.store(crate::portable::store_path(STORE_FILE))
        .map_err(|_| "Cannot open dictation preferences".to_owned())?;
    let value = serde_json::to_value(&config).map_err(|_| "Invalid preferences")?;
    store.set("config", value);
    store.save().map_err(|_| "Cannot save dictation preferences".to_owned())?;
    if let Some(secret) = token {
        *token_store().lock().map_err(|_| "Token store is locked")? =
            if secret.is_empty() { None } else { Some(secret) };
    }
    Ok(get_remote_dictation_status(app))
}

#[derive(Deserialize)]
struct ServerTranscription {
    raw_text: String,
    clean_text: Option<String>,
}

pub async fn transcribe(app: &AppHandle, samples: &[f32]) -> Result<Transcription, String> {
    let config = current_config(app);
    if config.route == DictationRoute::Local {
        return Err("Local route selected".to_owned());
    }
    let token = token_store().lock()
        .map_err(|_| "Token store is locked")?
        .clone()
        .ok_or("Remote server token is missing")?;
    let mut base = validate_url(&config.server_url)?;
    base.set_path("/v1/transcriptions");
    let estimated_bytes = samples.len().checked_mul(2)
        .and_then(|n| n.checked_add(44))
        .ok_or("Recording is too large")?;
    if estimated_bytes > MAX_WAV_BYTES {
        return Err("Recording exceeds the server's 25 MiB upload limit".into());
    }
    let mut output = Cursor::new(Vec::with_capacity(estimated_bytes));
    {
        let spec = hound::WavSpec {
            channels: 1, sample_rate: 16_000, bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut output, spec)
            .map_err(|_| "Failed to encode audio")?;
        for &value in samples {
            let pcm = (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(pcm).map_err(|_| "Failed to encode audio")?;
        }
        writer.finalize().map_err(|_| "Failed to finalize WAV")?;
    }
    let wav = output.into_inner();
    let form = multipart::Form::new()
        .part("audio", multipart::Part::bytes(wav)
            .file_name("dictation.wav")
            .mime_str("audio/wav").map_err(|_| "Cannot encode multipart audio")?)
        .text("cleanup", config.cleanup.to_string())
        .text("tone", config.tone);
    let mut form = form;
    let language = get_settings(app).selected_language;
    if !language.is_empty() && language != "auto" {
        form = form.text("language", language);
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(4))
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Cannot initialize remote client")?;
    let response = client.post(base).bearer_auth(token).multipart(form).send().await
        .map_err(|_| "Server connection or processing timeout")?;
    if !response.status().is_success() {
        return Err(format!("Server refused transcription (HTTP {})", response.status().as_u16()));
    }
    let payload: ServerTranscription = response.json().await
        .map_err(|_| "Invalid server transcription response")?;
    let text = payload.clean_text
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(payload.raw_text);
    Ok(Transcription { text, language: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_is_the_default() {
        assert_eq!(RemoteDictationConfig::default().route, DictationRoute::Local);
        assert!(!RemoteDictationConfig::default().cleanup);
    }

    #[test]
    fn reject_insecure_and_credentialed_server_urls() {
        for url in ["http://example.com", "http://localhost.evil.com",
                    "https://user:pass@example.com", "https://example.com/v1",
                    "https://example.com/?token=secret"] {
            assert!(validate_url(url).is_err(), "{url}");
        }
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://localhost:8765").is_ok());
    }
}
