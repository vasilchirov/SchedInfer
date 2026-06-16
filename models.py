from dataclasses import dataclass

@dataclass
class Request:
    id: int
    prompt_len: int
    response_len: int
    l_qo: int
    l_kv: int

@dataclass
class WorkloadChunk:
    req_id: int
    l_qo: int
    l_kv: int
    kv_start: int

@dataclass
class WorkerMetrics:
    worker_id: int
    iterations: int
    cache_misses: int
    latency_ns: int
    slack_ns: int

@dataclass
class EvaluationResult:
    makespan: int
    num_steps: int
    avg_step_latency: float
    throughput: float # tokens/s
    worker_metrics: list[WorkerMetrics]