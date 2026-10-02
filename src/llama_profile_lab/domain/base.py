"""Shared immutable-domain and content-identity primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_NON_SEMANTIC_KEYS = frozenset({"label", "description", "display_order"})


def _strip_non_semantic(value: Any, excluded: frozenset[str]) -> Any:
    """Recursively remove known presentation metadata from identity payloads."""
    if isinstance(value, Mapping):
        return {
            key: _strip_non_semantic(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, (list, tuple)):
        return [_strip_non_semantic(item, excluded) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class FrozenModel(BaseModel):
    """Base class for immutable, strict domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ContentAddressedModel(FrozenModel):
    """Immutable model with deterministic semantic content identity."""

    identity_exclude: ClassVar[frozenset[str]] = frozenset()

    def identity_payload(self) -> dict[str, Any]:
        """Return the semantic payload used for content identity."""
        payload = self.model_dump(mode="json", exclude_none=False)
        excluded = _NON_SEMANTIC_KEYS | self.identity_exclude
        stripped = _strip_non_semantic(payload, excluded)
        if not isinstance(stripped, dict):
            raise TypeError("content-addressed model payload must be an object")
        return stripped

    def canonical_identity_json(self) -> str:
        """Return deterministic JSON for the semantic identity payload."""
        return canonical_json(self.identity_payload())

    def content_hash(self) -> str:
        """Return SHA-256 over the semantic identity payload."""
        return sha256_json(self.identity_payload())
