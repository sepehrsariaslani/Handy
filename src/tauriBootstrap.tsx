import React from "react";
import ReactDOM from "react-dom/client";
import { platform } from "@tauri-apps/plugin-os";
import App from "./App";
import {
  applyTheme,
  getStoredTheme,
  syncThemeFromSettings,
} from "./lib/utils/theme";

// Native-only bootstrap: these modules call Tauri commands at startup.
import "./i18n";

import { useModelStore } from "./stores/modelStore";
import { useTranscriptionJobsStore } from "./stores/transcriptionJobsStore";

// Set platform before rendering so CSS can scope per-platform.
document.documentElement.dataset.platform = platform();

// Apply the last-known theme before first render, then sync from settings.
applyTheme(getStoredTheme());
syncThemeFromSettings();

useModelStore.getState().initialize();
useTranscriptionJobsStore.getState().initialize();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
