"""llama.cpp help capability parsing tests."""

from pathlib import Path

from llama_profile_lab.llama import (
    CapabilitySet,
    compare_capabilities,
    parse_help_options,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "llama"


def test_help_parser_preserves_short_and_long_options() -> None:
    text = (_FIXTURES / "llama-bench-help.txt").read_text(encoding="utf-8")
    options = parse_help_options(text)

    assert "--batch-size" in options
    assert "-b" in options
    assert "--ubatch-size" in options
    assert "-ub" in options
    assert "-pg" in options
    assert "--spec-type" not in options


def test_capability_sets_distinguish_server_only_flags() -> None:
    bench_text = (_FIXTURES / "llama-bench-help.txt").read_text(encoding="utf-8")
    server_text = (_FIXTURES / "custom-server-help.txt").read_text(encoding="utf-8")
    bench = CapabilitySet(
        kind="llama-bench",
        options=parse_help_options(bench_text),
        help_stdout=bench_text,
        help_stderr="",
        help_exit_code=0,
    )
    server = CapabilitySet(
        kind="llama-server",
        options=parse_help_options(server_text),
        help_stdout=server_text,
        help_stderr="",
        help_exit_code=0,
    )

    diff = compare_capabilities(bench, server)

    assert server.supports("--spec-type")
    assert server.supports("--spec-draft-n-max")
    assert not bench.supports("--spec-type")
    assert "--spec-type" in diff.only_right
    assert "--n-prompt" in diff.only_left


def test_fit_params_fixture_exposes_placement_capabilities() -> None:
    text = (_FIXTURES / "llama-fit-params-help.txt").read_text(encoding="utf-8")
    capabilities = CapabilitySet(
        kind="llama-fit-params",
        options=parse_help_options(text),
        help_stdout=text,
        help_stderr="",
        help_exit_code=0,
    )

    assert capabilities.supports("--ctx-size")
    assert capabilities.supports("--n-gpu-layers")
    assert capabilities.supports("--override-tensor")
    assert capabilities.supports("--fit-target")
    assert not capabilities.supports("--spec-type")
