"""Read-only adapter for llama-profile-launcher host configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_profile_lab.domain.base import JsonScalar


class LauncherProfileError(RuntimeError):
    """Raised when launcher profile configuration is unavailable or malformed."""


@dataclass(frozen=True, slots=True)
class LauncherProfile:
    id: str
    binary_key: str
    binary_path: str
    profiles: tuple[str, ...]
    model_path: str
    draft_model_path: str | None
    server_alias: str | None
    args: dict[str, JsonScalar]


class LauncherProfileProvider:
    """Resolve effective launcher model settings without mutating launcher config."""

    def __init__(self, config_path: Path | None) -> None:
        self.config_path = None if config_path is None else config_path.expanduser()

    @property
    def configured(self) -> bool:
        return self.config_path is not None

    def list(self) -> tuple[LauncherProfile, ...]:
        if self.config_path is None:
            return ()
        payload = self._load()
        models = _mapping(payload, "models")
        return tuple(
            self._resolve(payload, model_id)
            for model_id in sorted(models)
        )

    def get(self, model_id: str) -> LauncherProfile | None:
        if self.config_path is None:
            return None
        payload = self._load()
        models = _mapping(payload, "models")
        if model_id not in models:
            return None
        return self._resolve(payload, model_id)

    def _load(self) -> dict[str, Any]:
        path = self.config_path
        if path is None:
            raise LauncherProfileError("launcher profile configuration is not configured")
        if not path.is_file():
            raise LauncherProfileError(f"launcher profile config does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LauncherProfileError(
                f"cannot read launcher profile config {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LauncherProfileError("launcher profile config must be a JSON object")
        return payload

    def _resolve(
        self,
        payload: dict[str, Any],
        model_id: str,
    ) -> LauncherProfile:
        binaries = _mapping(payload, "binaries")
        profiles = _mapping(payload, "profiles")
        models = _mapping(payload, "models")
        raw_model = models.get(model_id)
        if not isinstance(raw_model, dict):
            raise LauncherProfileError(f"launcher model {model_id!r} must be an object")

        binary_key = _string(raw_model, "binary", context=model_id)
        binary_path = binaries.get(binary_key)
        if not isinstance(binary_path, str) or not binary_path:
            raise LauncherProfileError(
                f"launcher model {model_id!r} references unknown binary {binary_key!r}"
            )

        effective_args: dict[str, JsonScalar] = {}
        defaults = payload.get("defaults", {})
        if defaults is not None:
            if not isinstance(defaults, dict):
                raise LauncherProfileError("launcher defaults must be an object")
            effective_args.update(_args(defaults, "defaults"))

        raw_profile_names = raw_model.get("profiles", [])
        if not isinstance(raw_profile_names, list) or not all(
            isinstance(item, str) and item for item in raw_profile_names
        ):
            raise LauncherProfileError(
                f"launcher model {model_id!r} profiles must be a string list"
            )
        profile_names = tuple(raw_profile_names)
        for profile_name in profile_names:
            raw_profile = profiles.get(profile_name)
            if not isinstance(raw_profile, dict):
                raise LauncherProfileError(
                    f"launcher model {model_id!r} references unknown profile "
                    f"{profile_name!r}"
                )
            effective_args.update(_args(raw_profile, f"profile {profile_name!r}"))

        effective_args.update(_args(raw_model, f"model {model_id!r}"))

        model_path = _string(raw_model, "model", context=model_id)
        draft_raw = raw_model.get("draft_model")
        if draft_raw is not None and not isinstance(draft_raw, str):
            raise LauncherProfileError(
                f"launcher model {model_id!r} draft_model must be a string"
            )
        alias_raw = raw_model.get("server_alias")
        if alias_raw is not None and not isinstance(alias_raw, str):
            raise LauncherProfileError(
                f"launcher model {model_id!r} server_alias must be a string"
            )

        return LauncherProfile(
            id=model_id,
            binary_key=binary_key,
            binary_path=str(Path(binary_path).expanduser()),
            profiles=profile_names,
            model_path=str(Path(model_path).expanduser()),
            draft_model_path=(
                None if draft_raw is None else str(Path(draft_raw).expanduser())
            ),
            server_alias=alias_raw,
            args=effective_args,
        )


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise LauncherProfileError(f"launcher {key} must be an object")
    return value


def _string(mapping: dict[str, Any], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise LauncherProfileError(f"launcher {context!r} {key} must be a string")
    return value


def _args(mapping: dict[str, Any], context: str) -> dict[str, JsonScalar]:
    raw = mapping.get("args", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LauncherProfileError(f"launcher {context} args must be an object")
    result: dict[str, JsonScalar] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise LauncherProfileError(f"launcher {context} arg names must be strings")
        if value is not None and not isinstance(value, str | int | float | bool):
            raise LauncherProfileError(
                f"launcher {context} arg {key!r} must be a JSON scalar"
            )
        result[key] = value
    return result
