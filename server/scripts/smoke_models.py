"""Run after deployment to exercise REAL Whisper/Qwen workers, not CI mocks."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="fa")
    args = parser.parse_args()
    token = os.environ.get("HANDY_API_KEY", "")
    if len(token) < 24:
        parser.error("Export HANDY_API_KEY first")
    headers = {"Authorization": "Bearer " + token}
    with httpx.Client(base_url=args.server, headers=headers, timeout=180) as client:
        response = client.get("/v1/models/status")
        response.raise_for_status()
        status = response.json()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if not status["asr_ready"] or not status["llm_ready"]:
            parser.error("Both models must report readiness")
        start = perf_counter()
        result = client.post("/v1/text/cleanup", json={
            "text": "خب اِم، جلسه پارس‌فریم ساعت ۱۲ با هزینه ۱۵۰۰ تومان بود.",
            "tone": "professional", "language": "fa",
            "protected_terms": ["پارس‌فریم"],
        })
        result.raise_for_status()
        value = result.json()
        print("Editing latency ms:", round((perf_counter() - start) * 1000, 1))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.audio:
            if not args.audio.is_file():
                parser.error("Audio does not exist")
            start = perf_counter()
            with args.audio.open("rb") as stream:
                result = client.post(
                    "/v1/transcriptions",
                    files={"audio": (args.audio.name, stream, "application/octet-stream")},
                    data={"language": args.language, "cleanup": "true", "tone": "professional"},
                )
            result.raise_for_status()
            response = result.json()
            # Never print a user's private full transcript into logs.
            print(json.dumps({
                "language": response["language"],
                "model": response["model"],
                "segments": len(response["segments"]),
                "latency_ms": round((perf_counter() - start) * 1000, 1),
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
