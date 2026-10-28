"""Small development connectivity probe; never a reportable evaluation run.

Credentials come from HKUST_API_KEY or an explicitly supplied existing file.
No credential is copied to the output directory. No retries are made because a
failed request may already have consumed tokens.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superagent.llm_planner import NoRedirects, endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://aigc-api.hkust-gz.edu.cn/api/v2")
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    secret = os.environ.get("HKUST_API_KEY")
    if not secret and args.key_file:
        secret = args.key_file.read_text(encoding="utf-8-sig").strip()
    if not secret:
        raise ValueError("Set HKUST_API_KEY or supply --key-file")
    url = endpoint(args.base_url)
    args.output.mkdir(parents=True, exist_ok=False)
    body = {"model": args.model, "stream": False, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with OK only."}]}
    (args.output / "request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    report = {"purpose": "Development connectivity only; excluded from course evaluation results",
              "started_at": datetime.now(timezone.utc).isoformat(), "endpoint": url,
              "requested_model": args.model, "attempts": 1, "usage": None,
              "cost_rmb": None,
              "cost_status": "Unverified deployment version and 2026-10-21 official CNY rates; not assumed zero"}
    request = Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + secret})
    started = time.monotonic()
    try:
        with build_opener(NoRedirects()).open(request, timeout=60) as response:
            report["http_status"] = response.status
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Oversized API response")
        result = json.loads(raw)
        report.update(response_model=result.get("model"), usage=result.get("usage"),
                      choices=result.get("choices"), response_id=result.get("id"))
        report["status"] = "success" if result.get("choices") and isinstance(result.get("usage"), dict) else "incomplete_response"
    except HTTPError as exc:
        report.update(status="http_error", http_status=exc.code, error_type=type(exc).__name__)
    except Exception as exc:
        report.update(status="request_failed_usage_unknown", error_type=type(exc).__name__,
                      error_detail=str(getattr(exc, "reason", exc)))
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        rendered = json.dumps(report, ensure_ascii=False, indent=2).replace(secret, "<REDACTED>")
        (args.output / "result.json").write_text(rendered, encoding="utf-8")
        print(rendered)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
