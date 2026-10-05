from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("remotes.json")


class RemoteConfigFileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConfiguredRemote:
    name: str
    url: str
    enabled: bool = True
    email: str = ""
    password: str = ""
    queue_id: str = "default"
    timeout_seconds: float = 30.0
    probe_timeout_seconds: float = 2.5


def _as_nonempty_string(value: Any, field: str, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise RemoteConfigFileError(f"remotes[{index}].{field} must be a non-empty string")
    return text


def load_remotes(path: Path = CONFIG_PATH) -> list[ConfiguredRemote]:
    if not path.exists():
        raise RemoteConfigFileError(
            f"Remote configuration file was not found: {path}. "
            "Create remotes.json beside the node pack's Python files."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RemoteConfigFileError(
            f"Could not parse {path.name}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except Exception as exc:
        raise RemoteConfigFileError(f"Could not read {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("remotes"), list):
        raise RemoteConfigFileError(f"{path.name} must contain a top-level 'remotes' array")

    remotes: list[ConfiguredRemote] = []
    seen: set[str] = set()

    for index, entry in enumerate(raw["remotes"]):
        if not isinstance(entry, dict):
            raise RemoteConfigFileError(f"remotes[{index}] must be an object")

        name = _as_nonempty_string(entry.get("name"), "name", index)
        key = name.casefold()
        if key in seen:
            raise RemoteConfigFileError(f"Remote name '{name}' is duplicated")
        seen.add(key)

        url = _as_nonempty_string(entry.get("url"), "url", index).rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise RemoteConfigFileError(f"Remote '{name}' URL must start with http:// or https://")

        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RemoteConfigFileError(f"Remote '{name}' enabled must be true or false")

        queue_id = str(entry.get("queue_id") or "default").strip() or "default"

        try:
            timeout_seconds = float(entry.get("timeout_seconds", 30.0))
            probe_timeout_seconds = float(entry.get("probe_timeout_seconds", 2.5))
        except (TypeError, ValueError) as exc:
            raise RemoteConfigFileError(
                f"Remote '{name}' timeout_seconds/probe_timeout_seconds must be numbers"
            ) from exc

        if timeout_seconds <= 0 or probe_timeout_seconds <= 0:
            raise RemoteConfigFileError(f"Remote '{name}' timeouts must be greater than zero")

        remotes.append(
            ConfiguredRemote(
                name=name,
                url=url,
                enabled=enabled,
                email=str(entry.get("email") or "").strip(),
                password=str(entry.get("password") or ""),
                queue_id=queue_id,
                timeout_seconds=timeout_seconds,
                probe_timeout_seconds=probe_timeout_seconds,
            )
        )

    return remotes


def enabled_remotes(remotes: list[ConfiguredRemote]) -> list[ConfiguredRemote]:
    return [remote for remote in remotes if remote.enabled]


def find_remote(remotes: list[ConfiguredRemote], name: str) -> ConfiguredRemote | None:
    wanted = name.strip().casefold()
    if not wanted:
        return None
    return next((remote for remote in remotes if remote.name.casefold() == wanted), None)
