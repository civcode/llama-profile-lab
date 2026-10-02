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
from llama_profile_lab.llama.fit_params import (
    LlamaFitParamsAdapter,
    LlamaFitParamsConfigurationError,
    LlamaFitParamsParseError,
    LlamaFitParamsResult,
)
from llama_profile_lab.llama.server import (
    LlamaServerAdapter,
    LlamaServerConfigurationError,
)
from llama_profile_lab.llama.speed_bench import (
    SpeedBenchAdapter,
    SpeedBenchConfigurationError,
    SpeedBenchParseError,
    SpeedBenchResult,
    SpeedBenchSummary,
    parse_speed_bench_json,
)

__all__ = [
    "LlamaBenchAdapter",
    "LlamaBenchConfigurationError",
    "LlamaBenchParseError",
    "LlamaBenchResult",
    "LlamaBenchSample",
    "LlamaFitParamsAdapter",
    "LlamaFitParamsConfigurationError",
    "LlamaFitParamsParseError",
    "LlamaFitParamsResult",
    "LlamaServerAdapter",
    "LlamaServerConfigurationError",
    "SpeedBenchAdapter",
    "SpeedBenchConfigurationError",
    "SpeedBenchParseError",
    "SpeedBenchResult",
    "SpeedBenchSummary",
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
    "parse_speed_bench_json",
    "probe_binary",
    "sha256_file",
]
