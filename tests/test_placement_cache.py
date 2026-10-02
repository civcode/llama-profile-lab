"""Placement cache identity tests."""

from pathlib import Path

from llama_profile_lab.db.records import BinaryRecord
from llama_profile_lab.domain import Candidate, sha256_json
from llama_profile_lab.execution.placement import placement_cache_key
from tests.test_llama_fit_params import candidate


def fit_binary() -> BinaryRecord:
    return BinaryRecord(
        id="bin_fit",
        sha256="a" * 64,
        kind="llama-fit-params",
        path="/bin/llama-fit-params",
        size_bytes=123,
        mtime_ns=456,
        git_commit="abcdef1",
        git_branch=None,
        git_dirty=None,
        build_number="9000",
        build_info={},
        capabilities={},
        created_at="2026-01-01T00:00:00Z",
    )


def test_cache_key_ignores_non_fit_candidate_settings(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    base = candidate()
    payload = base.model_dump(mode="python")
    payload["server"]["parallel"] = 8
    changed = Candidate.model_validate(payload)

    left = placement_cache_key(
        candidate=base,
        hardware_fingerprint="host-a",
        fit_binary=fit_binary(),
        model_path=model,
    )
    right = placement_cache_key(
        candidate=changed,
        hardware_fingerprint="host-a",
        fit_binary=fit_binary(),
        model_path=model,
    )

    assert base.content_hash() != changed.content_hash()
    assert sha256_json(left) == sha256_json(right)


def test_cache_key_changes_for_fit_relevant_values(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    base = candidate()
    payload = base.model_dump(mode="python")
    payload["compute"]["ubatch_size"] = 1024
    changed = Candidate.model_validate(payload)

    left = placement_cache_key(
        candidate=base,
        hardware_fingerprint="host-a",
        fit_binary=fit_binary(),
        model_path=model,
    )
    right = placement_cache_key(
        candidate=changed,
        hardware_fingerprint="host-a",
        fit_binary=fit_binary(),
        model_path=model,
    )

    assert sha256_json(left) != sha256_json(right)


def test_cache_key_changes_for_host_or_fit_build(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    base = candidate()
    first_binary = fit_binary()
    second_binary = BinaryRecord(
        id="bin_fit_2",
        sha256="b" * 64,
        kind=first_binary.kind,
        path=first_binary.path,
        size_bytes=first_binary.size_bytes,
        mtime_ns=first_binary.mtime_ns,
        git_commit=first_binary.git_commit,
        git_branch=first_binary.git_branch,
        git_dirty=first_binary.git_dirty,
        build_number=first_binary.build_number,
        build_info=first_binary.build_info,
        capabilities=first_binary.capabilities,
        created_at=first_binary.created_at,
    )

    first = placement_cache_key(
        candidate=base,
        hardware_fingerprint="host-a",
        fit_binary=first_binary,
        model_path=model,
    )
    host_changed = placement_cache_key(
        candidate=base,
        hardware_fingerprint="host-b",
        fit_binary=first_binary,
        model_path=model,
    )
    binary_changed = placement_cache_key(
        candidate=base,
        hardware_fingerprint="host-a",
        fit_binary=second_binary,
        model_path=model,
    )

    assert sha256_json(first) != sha256_json(host_changed)
    assert sha256_json(first) != sha256_json(binary_changed)
