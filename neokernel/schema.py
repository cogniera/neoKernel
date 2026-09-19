"""Serializable evaluation records; all durations are in seconds unless marked ms."""

import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    S: int
    N: int

    def __post_init__(self):
        if min(self.batch, self.S, self.N) < 1:
            raise ValueError("workload dimensions must be positive")


PUBLIC = [Workload("public-0", 1, 512, 32), Workload("public-1", 4, 2048, 32),
          Workload("public-2", 16, 512, 128)]
GUESSED_HIDDEN = [Workload("h-a", 2, 1024, 64), Workload("h-b", 8, 1024, 64),
                  Workload("h-c", 32, 256, 64)]


def select_workloads(selection: str) -> list[Workload]:
    groups = {"public": PUBLIC, "hidden": GUESSED_HIDDEN, "all": PUBLIC + GUESSED_HIDDEN}
    if selection in groups:
        return list(groups[selection])
    names = selection.split(",")
    by_name = {w.name: w for w in groups["all"]}
    if not names or any(n not in by_name for n in names) or len(set(names)) != len(names):
        raise ValueError("unknown or duplicate workload name")
    return [by_name[n] for n in names]


@dataclass
class Correctness:
    passed: bool = True
    first_bad_position: list[int] | None = None
    margin: float | None = None
    near_tie_count: int = 0


@dataclass
class Sample:
    prompt: list[list[int]]
    tokens: list[list[int]]
    ttft_s: float
    tpot_s: float
    total_s: float
    peak_mem_bytes: int
    step_times_s: list[float] = field(default_factory=list)
    lifetime_peak_mem_bytes: int = 0
    pipe_overhead_ms: list[float] = field(default_factory=list)
    child_cpu_ms: list[float] = field(default_factory=list)
    child_diagnostics: dict = field(default_factory=dict)


@dataclass
class WorkloadResult:
    name: str
    batch: int
    S: int
    N: int
    samples: list[Sample] = field(default_factory=list)
    median_total: float = 0.0
    spread: float = 0.0
    tps: float = 0.0
    ttft_median: float = 0.0
    tpot_median: float = 0.0
    ttft_ratio: float | None = None
    tpot_ratio: float | None = None
    peak_mem_frac: float = 0.0
    correctness: Correctness = field(default_factory=Correctness)
    load_seconds: float = 0.0
    warmup_s: float = 0.0
    gates: dict[str, bool] = field(default_factory=dict)
    passed: bool = False
    failure_code: str | None = None
    note: str = ""
    floors: dict[str, Any] = field(default_factory=dict)
    pipe_overhead_ms: dict = field(default_factory=dict)
    cpu_step_ms: dict = field(default_factory=dict)
    measurement_order: str = ""


@dataclass
class RunResult:
    workloads: list[WorkloadResult]
    geomean_tps: float | None
    eligible: bool
    gpu_seconds: float = 0.0
    weight_bytes: int = 0
    parameter_count: int = 0
    protocol: str = "neokernel.random-prompts/1"
    native: dict = field(default_factory=dict)
    ts: str | None = None
    sha: str | None = None
    engine_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


PROPOSAL_FIELDS = ("item", "hypothesis", "expected_effect", "patch", "risk", "reasoning")
PROPOSAL_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in PROPOSAL_FIELDS},
                   "required": list(PROPOSAL_FIELDS), "additionalProperties": False}


@dataclass
class Proposal:
    item: str
    hypothesis: str
    expected_effect: str
    patch: str
    risk: str
    reasoning: str

    @classmethod
    def parse(cls, data: dict) -> "Proposal":
        if not isinstance(data, dict) or set(data) != set(PROPOSAL_FIELDS):
            raise ValueError("proposal must have exactly the schema fields")
        if any(not isinstance(v, str) or not v.strip() for v in data.values()):
            raise ValueError("proposal fields must be nonempty strings")
        return cls(**data)


@dataclass
class LogLine:
    id: int
    ts: str
    sha: str
    parent_sha: str | None
    proposer: str
    item: str
    hypothesis: str
    files_changed: list[str]
    guard: str
    workloads: list[dict]
    geomean_tps: float | None
    delta_pct: float | None
    kept: bool
    gpu_seconds: float
    note: str
    engine_sha256: str | None = None
    patch_sha256: str | None = None


def json_schema(annotation) -> dict:
    """Generate JSON schemas from the serialized dataclass shapes."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return {}
    if origin is types.UnionType:
        return {"anyOf": [json_schema(arg) for arg in arguments]}
    if origin is list or annotation is list:
        return {"type": "array", "items": json_schema(arguments[0]) if arguments else {}}
    if origin is dict or annotation is dict:
        return {"type": "object", "additionalProperties": json_schema(arguments[1]) if arguments else True}
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return {"type": "object", "properties": {f.name: json_schema(hints[f.name]) for f in fields(annotation)},
                "required": [f.name for f in fields(annotation)], "additionalProperties": False}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}[annotation]}


RUN_SCHEMA = json_schema(RunResult)
LOG_SCHEMA = json_schema(LogLine)
