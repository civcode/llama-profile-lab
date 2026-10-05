# llama-memory-estimator helper contract v1

Status: V2-M2 implementation contract

## Purpose

llama-profile-lab does not link its Python process against an arbitrary local llama.cpp build.
Per-device memory estimation is performed by an executable built alongside the llama.cpp build
being measured and registered independently by SHA-256.

The preferred llama.cpp primitive is common_get_device_memory_data(...), which reports total,
free, model, context, and compute memory per participating device. An equivalent supported
llama.cpp API is acceptable if the normalized output below is preserved.

## Executable identity

The auto-discovered executable name is `llama-memory-estimator`, registered with binary kind
`llama-memory-estimator`. Before execution or cache reuse, the current file SHA-256 is compared
with the registered hash. Drift fails closed and is persisted as a `binary_changed` attempt.

## Required command-line surface

The helper must advertise: `--json`, `--model`, `--ctx-size`, `--batch-size`,
`--ubatch-size`, `--cache-type-k`, `--cache-type-v`, `--gpu-layers`, `--split-mode`,
and `--main-gpu`. When required by the Candidate it must also advertise `--device`,
`--tensor-split`, `--override-tensor`, `--flash-attn`, `--load-mode`, `--lazy-mode`,
`--no-kv-offload`, `--no-op-offload`, `--no-host`, and `--no-repack`.

Selected-device order is semantic and must be preserved through execution and output.

## Output contract

With `--json`, stdout contains exactly one versioned JSON document; diagnostics belong on stderr.

~~~json
{
  "schema": "llama-memory-estimate",
  "version": 1,
  "devices": [
    {
      "logical_device_name": "CUDA0",
      "model_bytes": 0,
      "context_bytes": 0,
      "compute_bytes": 0,
      "total_bytes": 0,
      "device_total_bytes": 0,
      "device_free_bytes": 0
    }
  ],
  "resolved": {
    "n_gpu_layers": 0,
    "devices": ["CUDA0"],
    "split_mode": "layer",
    "main_gpu": 0,
    "tensor_split": null,
    "override_tensor": []
  },
  "metadata": {}
}
~~~

For every row, total_bytes equals model_bytes + context_bytes + compute_bytes, and free device
memory cannot exceed total device memory. Device rows cover exactly the logical devices in
`resolved.devices`.

## Failure and cache semantics

Nonzero exit, parser failure, timeout, cancellation, interruption, and helper hash drift have
distinct persisted attempt states with argv/stdout/stderr evidence.

The successful-estimate cache identity includes stable host identity, Candidate semantic hash,
model artifact identity, exact helper SHA-256, ordered selected devices, GPU-layer setting,
split mode, main GPU, tensor split, and tensor overrides.

## Persistence boundary

V2-M2 stores standalone estimates in `memory_estimate`, `memory_estimate_device`, and
`memory_estimate_attempt`. The existing `placement_device_memory` table requires a concrete
multi-model deployment placement. V2-M4 consumes M2 estimates and materializes selected
per-instance rows there once a joint deployment placement exists.

## Debug commands

~~~text
llprof binary devices BINARY_ID
llprof placement estimate CANDIDATE_ID --helper-binary HELPER_BINARY_ID \
  --model-path MODEL.gguf --device CUDA0 --device Vulkan0
~~~
