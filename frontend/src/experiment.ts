import type {
  BinaryRecord,
  Candidate,
  JsonScalar,
  LauncherProfile,
  ParameterDefinition,
  SearchDimension,
  WorkloadSuiteCase
} from "./types";

function numberArg(profile: LauncherProfile, name: string, fallback: number): number {
  const value = profile.args[name];
  return typeof value === "number" ? value : fallback;
}

function stringArg(profile: LauncherProfile, name: string, fallback: string): string {
  const value = profile.args[name];
  return typeof value === "string" ? value : fallback;
}

function gpuLayers(profile: LauncherProfile): number | "auto" | "all" | null {
  const value = profile.args["--n-gpu-layers"];
  if (typeof value === "number" || value === "auto" || value === "all") {
    return value;
  }
  return null;
}

export function candidateFromProfile(profile: LauncherProfile): Candidate {
  const specType = profile.args["--spec-type"];
  const draftN = profile.args["--spec-draft-n-max"];
  const speculative = typeof specType === "string" && typeof draftN === "number";
  return {
    schema: "llama-profile-candidate",
    version: 1,
    model: {
      target_model_id: "launcher-profile:" + profile.id,
      draft_model_id: profile.draft_model_path
        ? "launcher-draft:" + profile.id
        : null
    },
    context: {
      size: numberArg(profile, "--ctx-size", 4096),
      cache_type_k: stringArg(profile, "--cache-type-k", "f16"),
      cache_type_v: stringArg(profile, "--cache-type-v", "f16"),
      kv_offload: profile.args["--no-kv-offload"] !== true,
      kv_unified: profile.args["--no-kv-unified"] !== true
    },
    compute: {
      flash_attn: stringArg(profile, "--flash-attn", "auto") as
        | "on"
        | "off"
        | "auto",
      batch_size: numberArg(profile, "--batch-size", 2048),
      ubatch_size: numberArg(profile, "--ubatch-size", 512),
      threads:
        typeof profile.args["--threads"] === "number"
          ? Number(profile.args["--threads"])
          : null,
      load_mode: stringArg(profile, "--load-mode", "auto"),
      lazy_mode: stringArg(profile, "--lazy-mode", "auto"),
      repack: profile.args["--no-repack"] !== true,
      no_host: profile.args["--no-host"] === true,
      no_op_offload: profile.args["--no-op-offload"] === true
    },
    placement: {
      mode: "fit",
      fit: {
        target_mib: numberArg(profile, "--fit-target", 256),
        min_context: Math.min(4096, numberArg(profile, "--ctx-size", 4096))
      },
      constraints: {
        n_gpu_layers: gpuLayers(profile),
        n_cpu_moe: numberArg(profile, "--n-cpu-moe", 0),
        split_mode: stringArg(profile, "--split-mode", "layer"),
        main_gpu: numberArg(profile, "--main-gpu", 0),
        devices: "auto",
        tensor_split: null,
        override_tensor: []
      }
    },
    server: {
      parallel: numberArg(profile, "--parallel", 1)
    },
    speculative: speculative
      ? {
          enabled: true,
          type: String(specType),
          draft_n_max: Number(draftN)
        }
      : {
          enabled: false,
          type: null,
          draft_n_max: null
        },
    extra_args: {}
  };
}

export function parseDimensionValues(
  raw: string,
  definition: ParameterDefinition
): JsonScalar[] {
  const pieces = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!pieces.length) {
    throw new Error(definition.label + " needs at least one value.");
  }
  return pieces.map((piece) => {
    if (definition.value_types.includes("bool")) {
      if (piece === "true") return true;
      if (piece === "false") return false;
    }
    if (
      definition.value_types.includes("int") ||
      definition.value_types.includes("float")
    ) {
      const numeric = Number(piece);
      if (Number.isFinite(numeric)) return numeric;
    }
    if (piece === "null" && definition.value_types.includes("null")) return null;
    return piece;
  });
}

export function binaryOptions(binary: BinaryRecord | null): Set<string> {
  if (!binary) return new Set();
  const raw = binary.capabilities.options;
  if (!Array.isArray(raw)) return new Set();
  return new Set(raw.filter((value): value is string => typeof value === "string"));
}

export function dimensionSupported(
  definition: ParameterDefinition,
  binary: BinaryRecord | null
): boolean {
  if (!binary || !definition.cli_argument) return true;
  if (!definition.supported_by.includes(binary.kind)) return true;
  const options = binaryOptions(binary);
  return options.size === 0 || options.has(definition.cli_argument);
}

function pathValue(candidate: Candidate, path: string): JsonScalar {
  let current: unknown = candidate;
  for (const segment of path.split(".")) {
    if (!current || typeof current !== "object" || !(segment in current)) {
      return null;
    }
    current = (current as Record<string, unknown>)[segment];
  }
  return typeof current === "string" ||
    typeof current === "number" ||
    typeof current === "boolean" ||
    current === null
    ? current
    : null;
}

function relation(
  left: JsonScalar,
  operator: string,
  right: JsonScalar
): boolean {
  if (operator === "==") return left === right;
  if (operator === "!=") return left !== right;
  if (typeof left !== "number" || typeof right !== "number") return true;
  if (operator === "<=") return left <= right;
  if (operator === ">=") return left >= right;
  if (operator === "<") return left < right;
  if (operator === ">") return left > right;
  return true;
}

function constraintMatches(
  candidate: Candidate,
  values: Record<string, JsonScalar>,
  expression: string
): boolean {
  const match = expression.match(
    /^\s*([A-Za-z0-9_.]+)\s*(<=|>=|==|!=|<|>)\s*([A-Za-z0-9_.]+)\s*$/
  );
  if (!match) return true;
  const [, leftPath, operator, rightPath] = match;
  const left = values[leftPath] ?? pathValue(candidate, leftPath);
  const right = values[rightPath] ?? pathValue(candidate, rightPath);
  return relation(left, operator, right);
}

export function previewCandidateCount(
  candidate: Candidate,
  dimensions: SearchDimension[],
  constraints: string[]
): { raw: number; valid: number; rejected: number } {
  let points: Record<string, JsonScalar>[] = [{}];
  for (const dimension of dimensions) {
    points = points.flatMap((point) =>
      dimension.values.map((value) => ({ ...point, [dimension.path]: value }))
    );
  }
  const valid = points.filter((point) =>
    constraints.every((constraint) =>
      constraintMatches(candidate, point, constraint)
    )
  ).length;
  return { raw: points.length, valid, rejected: points.length - valid };
}

export function defaultWorkloads(): WorkloadSuiteCase[] {
  return [
    {
      kind: "microbench-prefill",
      label: "PP2K @ d0",
      safety_margin_tokens: 0,
      prompt_tokens: 2048,
      depth: { type: "absolute", tokens: 0 }
    },
    {
      kind: "microbench-prefill",
      label: "PP8K @ d0",
      safety_margin_tokens: 0,
      prompt_tokens: 8192,
      depth: { type: "absolute", tokens: 0 }
    },
    {
      kind: "microbench-decode",
      label: "TG256 @ 4K",
      safety_margin_tokens: 0,
      generate_tokens: 256,
      depth: { type: "absolute", tokens: 4096 }
    },
    {
      kind: "microbench-decode",
      label: "TG256 @ 50%",
      safety_margin_tokens: 0,
      generate_tokens: 256,
      depth: { type: "fraction", value: 0.5 }
    }
  ];
}

export function workloadLabel(item: WorkloadSuiteCase, index: number): string {
  if (item.label) return item.label;
  if (item.kind === "speed-bench") return "Server workload " + (index + 1);
  if (item.kind === "microbench-prefill") {
    return "PP" + item.prompt_tokens;
  }
  if (item.kind === "microbench-decode") {
    return "TG" + item.generate_tokens;
  }
  return "PP" + item.prompt_tokens + " + TG" + item.generate_tokens;
}

export function workloadFilter(index: number): string {
  return "suite_case_index=" + index;
}
