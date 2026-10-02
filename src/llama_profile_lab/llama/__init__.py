"""llama.cpp adapters and capability discovery."""

from llama_profile_lab.llama.bench import (
    LlamaBenchAdapter,
    LlamaBenchConfigurationError,
    LlamaBenchParseError,
    LlamaBenchResult,
    LlamaBenchSample,
    parse_llama_bench_json,
)
from llama_profile_lab.llama.capabilities import (
    BinaryKind,
    CapabilityDiff,
    CapabilitySet,
    compare_capabilities,
    parse_help_options,
)
from llama_profile_lab.llama.discovery import (
    BinaryDiscoveryError,
    BinaryProbe,
    CommandCapture,
    discover_binary_paths,
    infer_binary_kind,
    parse_build_metadata,
    probe_binary,
    sha256_file,
)

__all__ = [
    "LlamaBenchAdapter",
    "LlamaBenchConfigurationError",
    "LlamaBenchParseError",
    "LlamaBenchResult",
    "LlamaBenchSample",
    "BinaryDiscoveryError",
    "BinaryKind",
    "BinaryProbe",
    "CapabilityDiff",
    "CapabilitySet",
    "CommandCapture",
    "compare_capabilities",
    "discover_binary_paths",
    "infer_binary_kind",
    "parse_build_metadata",
    "parse_help_options",
    "parse_llama_bench_json",
    "probe_binary",
    "sha256_file",
]
