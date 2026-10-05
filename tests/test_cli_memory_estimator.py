"""CLI coverage for V2 device inventory and memory estimation."""

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main
from llama_profile_lab.db import CandidateRepository, Database, EnvironmentRepository
from tests.test_memory_estimator_service import make_candidate, write_helper


def test_cli_device_inventory_and_memory_estimate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    helper = tmp_path / "llama-memory-estimator"
    write_helper(helper)
    database_path = tmp_path / "benchmarks.db"

    assert main(
        [
            "binary",
            "inspect",
            str(helper),
            "--database",
            str(database_path),
        ]
    ) == 0

    with Database(database_path).session() as connection:
        binaries = EnvironmentRepository(connection).list_binaries()
        assert len(binaries) == 1
        helper_id = binaries[0].id
        candidate_id = CandidateRepository(connection).put(make_candidate())

    capsys.readouterr()
    assert main(
        [
            "binary",
            "devices",
            helper_id,
            "--database",
            str(database_path),
        ]
    ) == 0
    device_output = capsys.readouterr().out
    assert "CUDA0" in device_output
    assert "Vulkan0" in device_output
    assert "physical=unresolved" in device_output

    assert main(
        [
            "placement",
            "estimate",
            candidate_id,
            "--helper-binary",
            helper_id,
            "--model-path",
            str(tmp_path / "model.gguf"),
            "--device",
            "CUDA0",
            "--device",
            "Vulkan0",
            "--database",
            str(database_path),
        ]
    ) == 0
    estimate_output = capsys.readouterr().out
    assert "Cache: miss" in estimate_output
    assert "Devices: CUDA0,Vulkan0" in estimate_output
    assert "model=" in estimate_output
    assert "context=" in estimate_output
    assert "compute=" in estimate_output

    assert main(
        [
            "placement",
            "estimate",
            candidate_id,
            "--helper-binary",
            helper_id,
            "--model-path",
            str(tmp_path / "model.gguf"),
            "--device",
            "CUDA0",
            "--device",
            "Vulkan0",
            "--database",
            str(database_path),
        ]
    ) == 0
    assert "Cache: hit" in capsys.readouterr().out
