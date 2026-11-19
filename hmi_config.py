"""Shared configuration and credential-free OpenAI-compatible HTTP transport.

No third-party packages are required. Importing this module never calls an API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(os.environ.get("HMI_ROOT", Path(__file__).resolve().parent)).resolve()
CONFIG_FILE = Path(os.environ.get("HMI_CONFIG", ROOT / "paper_config.json"))
with CONFIG_FILE.open(encoding="utf-8") as stream:
    PAPER = json.load(stream)
PREPARED = ROOT / "prepared"
RUN_ID = os.environ.get("HMI_RUN_ID", "default")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", RUN_ID) or ".." in RUN_ID:
    raise ValueError("HMI_RUN_ID must be a short filename-safe identifier.")
RUN_DIR = ROOT / "results" / RUN_ID
STAGE1 = RUN_DIR / "stage1"
STAGE2 = RUN_DIR / "stage2"
MODE = os.environ.get("HMI_RUN_MODE", "inspect")
SEED = int(PAPER["seed"])


def data_path(section: str, name: str) -> Path:
    value = Path(PAPER[section][name]).expanduser()
    return value if value.is_absolute() else ROOT / value


def model_name(role: str = "worker") -> str:
    prefix = {"worker": "HMI", "detector": "HMI_DETECTOR", "judge": "HMI_EVAL"}[role]
    return os.environ.get(prefix + "_MODEL") or os.environ.get("HMI_MODEL", "")


def jsonl_read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def jsonl_write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


@dataclass(frozen=True)
class ModelSettings:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout: int = 120
    retries: int = 3

    @classmethod
    def from_env(cls, role: str = "worker"):
        prefix = {"worker": "HMI", "detector": "HMI_DETECTOR", "judge": "HMI_EVAL"}[role]
        return cls(
            model=model_name(role),
            base_url=os.environ.get(prefix + "_BASE_URL") or os.environ.get("HMI_BASE_URL", ""),
            api_key=os.environ.get(prefix + "_API_KEY") or os.environ.get("HMI_API_KEY", ""),
            timeout=int(os.environ.get("HMI_TIMEOUT", "120")),
            retries=int(os.environ.get("HMI_RETRIES", "3")),
        )

    def validate(self):
        if not self.model or not self.base_url or not self.api_key:
            raise RuntimeError(
                "Set HMI_MODEL, HMI_BASE_URL and HMI_API_KEY in the environment; "
                "detector/judge overrides are optional. Do not put credentials in source files."
            )
        parsed = urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API URLs must not contain embedded credentials, queries or fragments.")
        if not parsed.hostname or parsed.scheme not in ("https", "http"):
            raise ValueError("Invalid HMI_BASE_URL.")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Use HTTPS for remote API endpoints.")
        if self.retries < 1 or self.timeout < 1:
            raise ValueError("HMI_RETRIES and HMI_TIMEOUT must be positive.")

    def public_metadata(self):
        return {"model": self.model, "base_url": self.base_url,
                "timeout": self.timeout, "retries": self.retries}


def redact_credentials(text: str) -> str:
    for name, value in os.environ.items():
        if name.startswith("HMI") and name.endswith("_API_KEY") and value:
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", text)


def chat_text(system: str, user: str, *, role="worker", json_mode=False) -> str:
    settings = ModelSettings.from_env(role)
    settings.validate()
    payload = {
        "model": settings.model, "temperature": 0, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        settings.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + settings.api_key},
        method="POST",
    )
    for attempt in range(settings.retries):
        try:
            with urlopen(request, timeout=settings.timeout) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return redact_credentials(str(content))
        except HTTPError as error:
            status = error.code
            error.close()
            if status not in (429, 500, 502, 503, 504) or attempt == settings.retries - 1:
                raise RuntimeError(f"Model API HTTP {status}; response body omitted.") from None
        except (URLError, TimeoutError):
            if attempt == settings.retries - 1:
                raise RuntimeError("Model API connection failed; endpoint details omitted.") from None
        time.sleep(attempt + 1)
    raise RuntimeError("Model API retries exhausted.")


def chat_json(system: str, user: str, *, role="worker") -> dict:
    text = chat_text(system, user, role=role).strip()
    fence = re.fullmatch(chr(96)*3 + r"(?:json)?\s*(.*?)\s*" + chr(96)*3, text, re.S | re.I)
    value = json.loads(fence.group(1) if fence else text)
    if not isinstance(value, dict):
        raise ValueError("The model must return a JSON object.")
    return value


def preflight(role="worker"):
    if MODE == "real":
        ModelSettings.from_env(role).validate()
        record_run_configuration()


def record_run_configuration():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for role in ("worker", "detector", "judge"):
        ModelSettings.from_env(role).validate()
    info = {"paper_config": PAPER,
            "models": {role: ModelSettings.from_env(role).public_metadata()
                       for role in ("worker", "detector", "judge")}}
    destination = RUN_DIR / "run_configuration.json"
    if destination.exists() and json.loads(destination.read_text()) != info:
        raise ValueError("This HMI_RUN_ID belongs to a different configuration. Use a new run ID.")
    destination.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def source_label(path: Path) -> str:
    """Persist relative dataset provenance, not a user's machine-specific path."""
    path=Path(path).resolve()
    roots=[("root",ROOT),("meditod",data_path("stage2","meditod_directory")),
           ("kddres",data_path("stage2","kddres_directory"))]
    for label,base in roots:
        try:
            relative=path.relative_to(base.resolve())
            return str(relative) if label=="root" else label+"/"+str(relative)
        except ValueError:
            pass
    return path.name
