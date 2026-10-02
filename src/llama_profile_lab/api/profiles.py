"""Read-only adapter for llama-profile-launcher host configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
    PlacementConstraints,
    ServerConfig,
    SpeculativeConfig,
)
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
    candidate: Candidate


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
            candidate=_candidate_from_args(
                model_id=model_id,
                has_draft_model=draft_raw is not None,
                args=effective_args,
            ),
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


def _candidate_from_args(
    *,
    model_id: str,
    has_draft_model: bool,
    args: dict[str, JsonScalar],
) -> Candidate:
    """Map launcher performance settings into the canonical Candidate schema."""
    spec_type = _optional_string_arg(args, "--spec-type")
    draft_n = _optional_int_arg(args, "--spec-draft-n-max")
    speculative = spec_type is not None and draft_n is not None

    return Candidate(
        model=ModelSelection(
            target_model_id=f"launcher-profile:{model_id}",
            draft_model_id=(
                f"launcher-draft:{model_id}" if has_draft_model else None
            ),
        ),
        context=ContextConfig(
            size=_int_arg(args, "--ctx-size", 4096),
            cache_type_k=_string_arg(args, "--cache-type-k", "f16"),
            cache_type_v=_string_arg(args, "--cache-type-v", "f16"),
            kv_offload=not _truthy_arg(args, "--no-kv-offload"),
            kv_unified=not _truthy_arg(args, "--no-kv-unified"),
        ),
        compute=ComputeConfig(
            flash_attn=_flash_attn(args),
            batch_size=_int_arg(args, "--batch-size", 2048),
            ubatch_size=_int_arg(args, "--ubatch-size", 512),
            threads=_optional_int_arg(args, "--threads"),
            load_mode=_string_arg(args, "--load-mode", "auto"),
            lazy_mode=_string_arg(args, "--lazy-mode", "auto"),
            repack=not _truthy_arg(args, "--no-repack"),
            no_host=_truthy_arg(args, "--no-host"),
            no_op_offload=_truthy_arg(args, "--no-op-offload"),
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(
                target_mib=_int_arg(args, "--fit-target", 256),
                min_context=min(4096, _int_arg(args, "--ctx-size", 4096)),
            ),
            constraints=PlacementConstraints(
                n_gpu_layers=_gpu_layers(args),
                n_cpu_moe=_int_arg(args, "--n-cpu-moe", 0),
                split_mode=_string_arg(args, "--split-mode", "layer"),
                main_gpu=_int_arg(args, "--main-gpu", 0),
            ),
        ),
        server=ServerConfig(parallel=_int_arg(args, "--parallel", 1)),
        speculative=(
            SpeculativeConfig(
                enabled=True,
                type=spec_type,
                draft_n_max=draft_n,
            )
            if speculative
            else SpeculativeConfig()
        ),
    )


def _int_arg(args: dict[str, JsonScalar], name: str, default: int) -> int:
    value = args.get(name)
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def _optional_int_arg(args: dict[str, JsonScalar], name: str) -> int | None:
    value = args.get(name)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _string_arg(args: dict[str, JsonScalar], name: str, default: str) -> str:
    value = args.get(name)
    return value if isinstance(value, str) and value else default


def _optional_string_arg(args: dict[str, JsonScalar], name: str) -> str | None:
    value = args.get(name)
    return value if isinstance(value, str) and value else None


def _truthy_arg(args: dict[str, JsonScalar], name: str) -> bool:
    value = args.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int | float):
        return value != 0
    return False


def _flash_attn(args: dict[str, JsonScalar]) -> str:
    value = _string_arg(args, "--flash-attn", "auto")
    return value if value in {"on", "off", "auto"} else "auto"


def _gpu_layers(args: dict[str, JsonScalar]) -> int | str | None:
    value = args.get("--n-gpu-layers")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if value in {"auto", "all"}:
        return str(value)
    return None
