# Handy phase 4 — main dictation shortcut: local, server, and offline fallback

## What has actually been integrated

The existing Rust global shortcut pipeline still owns microphone recording,
WAV persistence, transcription history and clipboard/paste. Instead of calling
only the local transcription model after recording stops, it now routes the
captured 16 kHz audio based on a user-selected preference:

| Mode                            | On shortcut start                                     | On shortcut stop                                                    | On server error                                                             |
| ------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Local                           | Preload local model and VAD                           | Existing transcribe-cpp implementation                              | Not applicable                                                              |
| Server first + offline fallback | Preload the local model and VAD                       | Upload WAV to authenticated Handy phase-2 API and use returned text | Reuse the _same captured audio_ with local transcribe-cpp                   |
| Server only                     | Record without demanding any local model be installed | Upload WAV to authenticated Handy phase-2 API                       | Return a visible transcription failure without silently using another model |

The resulting text continues through Handy's established optional
post-processing action, history recording, paste and cancellation checks. The
post-processing global shortcut still invokes its existing configured provider
after transcription; enabling Qwen cleanup in the server preferences can result
in an additional cleanup pass.

## How to use it

1. Launch the Handy desktop application and open **Live Dictation**.
2. In **Global shortcut dictation**, select the processing route.
3. Enter the API root URL, e.g. `http://127.0.0.1:8765` on the same machine,
   or a properly TLS-terminated `https://your-private-host`.
4. Enter the bearer key that matches `HANDY_API_KEY`, choose Qwen cleanup/tone
   and select **Save shortcut settings**. The token remains in process memory
   and **must be re-entered after restarting Handy**.
5. Use the existing global shortcut (`Option+Space` on default macOS, or
   `Ctrl+Space` on default Windows). Recording stops via the usual
   hold/toggle settings. The audio is sent after capture ends, and its
   transcript is pasted to the previously focused app by the existing path.
6. For offline fallback, the local speech model must be installed in Handy
   before selecting **Server first + offline fallback**.

You can continue using the **Live Dictation** separate-page real-time
WebSocket prototype introduced in phase 3. That page and the global shortcut
are not simultaneously recording: they use different capture backends and
should not be used together.

## Security, defaults and limits

- **Local mode is the default.** The desktop never uploads dictated audio
  unless the user explicitly changes to a server route.
- The API token is stored in an in-memory Rust mutex, not the settings store.
  Only server origin, route, cleanup and tone persist. It is never included in
  a URL, returned to frontend callers, or written to logs.
- Remote connections must be HTTPS except for localhost/loopback HTTP;
  credentials, query parameters and subpaths in the URL are rejected.
  HTTP redirects are disabled, with 4-second connection and 60-second
  overall request timeouts.
- The REST endpoint accepts short, bounded WAV recordings, maximum **25 MiB**
  including the WAV header. In fallback mode, overlong recordings route to
  the local model; in server-only mode they produce an error.
- Existing history still preserves the captured WAV locally. When Qwen
  cleanup is enabled the text returned by the server (rather than the raw
  intermediate transcript) is saved as the history transcription. The server
  itself returns both raw and clean fields.
- Remote-only does not download/load a local ASR model. No automatic
  substitution happens if the server is down.

## Verification

Run:

- `cd src-tauri && cargo fmt -- --check && cargo test remote_dictation`
- `bun run lint && bun run format:check && bun run build`

Manual verification needs a running server with Whisper Large V3:

- Select local mode, dictate, and verify typing/history unchanged.
- Select server-first with local model installed; dictate once with the
  server working and once after stopping the server. Verify both paste text.
- Select server-only and disconnect the network. Verify that the app reports
  transcription failed, not a false success.
- Check that a token is absent on restart, but the non-secret route persists.
- Test both main and post-processing shortcuts, paste cancellation, and
  microphone selection on macOS and Windows.

## Remaining limitations

**This is an after-stop route for the global keyboard shortcut.** True
word-by-word in-recording server updates, remote editing of Handy's existing
live overlay, and bidirectional WebSocket capture from the Rust recorder have
not been integrated. The independent WebView Live page implements chunked
streaming and on-screen partial/final text, but is not yet the underlying
system-wide keyboard-capture route.

The full desktop signing/build pipelines and real speech model/GPU latency
must be checked on deployment hardware; green CI does not substitute for
microphone, keyboard, clipboard, and server integration tests.
