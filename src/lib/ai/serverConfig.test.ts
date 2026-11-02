import assert from "node:assert/strict";
import { normalizeServerUrl } from "./serverConfig";

assert.equal(
  normalizeServerUrl("http://127.0.0.1:8765/"),
  "http://127.0.0.1:8765",
);
assert.equal(
  normalizeServerUrl("https://api.example.com/"),
  "https://api.example.com",
);
assert.equal(
  normalizeServerUrl("http://localhost:8765"),
  "http://localhost:8765",
);

for (const unsafe of [
  "http://example.com",
  "http://localhost.evil.example",
  "https://user:password@example.com",
  "https://example.com/v1",
  "https://example.com?token=secret",
  "ftp://example.com",
]) {
  assert.throws(() => normalizeServerUrl(unsafe), unsafe);
}
console.log("serverConfig: all assertions passed");
