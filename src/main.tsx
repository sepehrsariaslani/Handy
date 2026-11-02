import { isTauri } from "@tauri-apps/api/core";
import { installCompatShims } from "./lib/compat";

installCompatShims();

if (isTauri()) {
  // Load native modules only inside the Tauri WebView. The ordinary Vite URL
  // has no IPC bridge; importing i18n/stores there would call invoke() and
  // crash before an explanation could be shown.
  void import("./tauriBootstrap");
} else {
  const root = document.getElementById("root");
  if (root) {
    const panel = document.createElement("main");
    panel.style.cssText =
      "font-family:system-ui,sans-serif;max-width:640px;margin:15vh auto;padding:32px;line-height:1.7;color:inherit";
    const heading = document.createElement("h1");
    heading.textContent = "Open Handy as a desktop app";
    const explanation = document.createElement("p");
    explanation.textContent =
      "This URL is only Vite's development server. It cannot access the native microphone, model manager or Tauri commands.";
    const command = document.createElement("code");
    command.textContent = "bun run dev:mac";
    const hint = document.createElement("p");
    hint.textContent =
      "Run the command in a terminal from the Handy repository and use the native Handy Dev window, not this browser tab.";
    panel.append(heading, explanation, command, hint);
    root.replaceChildren(panel);
  }
}
