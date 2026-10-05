"""Executable discovery, probing, hashing, and persistence tests."""

from pathlib import Path

from llama_profile_lab.db import Database, EnvironmentRepository
from llama_profile_lab.llama import (
    CapabilitySet,
    compare_capabilities,
    discover_binary_paths,
    probe_binary,
)


def write_fake_binary(
    path: Path,
    *,
    version: str,
    help_text: str,
) -> None:
    script = f"""#!/usr/bin/env python3
import sys

if "--version" in sys.argv:
    print({version!r})
elif "--help" in sys.argv or "-h" in sys.argv:
    print({help_text!r})
else:
    raise SystemExit(2)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def persist_probe(database: Database, path: Path) -> str:
    probe = probe_binary(path)
    with database.session() as connection:
        return EnvironmentRepository(connection).put_binary(
            sha256=probe.sha256,
            kind=probe.kind,
            path=str(probe.path),
            size_bytes=probe.size_bytes,
            mtime_ns=probe.mtime_ns,
            git_commit=probe.git_commit,
            build_number=probe.build_number,
            build_info=probe.build_info_mapping(),
            capabilities=probe.capabilities.to_mapping(),
        )


def test_probe_hashes_exact_binary_and_parses_metadata(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    write_fake_binary(
        binary,
        version="version: 8123 (abcdef123)",
        help_text="  --batch-size N  batch\n  --spec-type TYPE  speculative",
    )

    probe = probe_binary(binary)

    assert probe.kind == "llama-server"
    assert len(probe.sha256) == 64
    assert probe.build_number == "8123"
    assert probe.git_commit == "abcdef123"
    assert probe.capabilities.supports("--batch-size")
    assert probe.capabilities.supports("--spec-type")


def test_native_and_custom_builds_remain_distinct_by_hash(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    qwen_dir = tmp_path / "qwen"
    native_dir.mkdir()
    qwen_dir.mkdir()

    native = native_dir / "llama-server"
    custom = qwen_dir / "llama-server"
    write_fake_binary(
        native,
        version="version: 8000 (aaaaaaaa)",
        help_text="  --batch-size N  batch\n",
    )
    write_fake_binary(
        custom,
        version="version: 8001 (bbbbbbbb)",
        help_text="  --batch-size N  batch\n  --spec-type TYPE  speculative\n",
    )

    database = Database(tmp_path / "benchmarks.db")
    native_id = persist_probe(database, native)
    custom_id = persist_probe(database, custom)

    assert native_id != custom_id

    with database.session() as connection:
        records = EnvironmentRepository(connection).list_binaries()
        assert len(records) == 2
        by_id = {record.id: record for record in records}
        native_caps = CapabilitySet.from_mapping(
            dict(by_id[native_id].capabilities),
            fallback_kind="llama-server",
        )
        custom_caps = CapabilitySet.from_mapping(
            dict(by_id[custom_id].capabilities),
            fallback_kind="llama-server",
        )

    diff = compare_capabilities(native_caps, custom_caps)
    assert "--spec-type" in diff.only_right


def test_discovery_searches_explicit_directories(tmp_path: Path) -> None:
    native_dir = tmp_path / "build" / "bin"
    native_dir.mkdir(parents=True)
    bench = native_dir / "llama-bench"
    fit = native_dir / "llama-fit-params"
    memory = native_dir / "llama-memory-estimator"
    server = native_dir / "llama-server"
    speed = native_dir / "speed_bench.py"
    write_fake_binary(bench, version="version: 1 (aaaaaaa)", help_text="  --model F")
    write_fake_binary(fit, version="version: 1 (aaaaaaa)", help_text="  --fit-target N")
    write_fake_binary(
        memory,
        version="version: 1 (aaaaaaa)",
        help_text="  --json\n  --list-devices",
    )
    write_fake_binary(server, version="version: 1 (aaaaaaa)", help_text="  --model F")
    write_fake_binary(speed, version="version: 1", help_text="  --url URL\n  --output FILE")

    discovered = discover_binary_paths((native_dir,), include_path=False)

    assert discovered == tuple(
        sorted(
            (
                bench.resolve(),
                fit.resolve(),
                memory.resolve(),
                server.resolve(),
                speed.resolve(),
            ),
            key=str,
        )
    )
    assert probe_binary(memory).kind == "llama-memory-estimator"
    assert probe_binary(speed).kind == "speed-bench"
