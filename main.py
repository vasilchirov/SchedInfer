from simulator import Simulator
from transformers import AutoTokenizer
from data_processing import DataProcessor
from schedules import flash_attention, flash_decoding, lean_attention, flashinfer, flashinfer_kk
from models import EvaluationResult


def print_result(name: str, result: EvaluationResult) -> None:
    col_w = [6, 12, 14, 14, 12, 8]
    headers = ["Worker", "Iterations", "Cache Misses", "Latency (ns)", "Slack (ns)", "Load %"]

    title = f" {name} "
    total_w = sum(col_w) + len(col_w) * 3 + 1
    bar = "=" * total_w

    print(f"\n{bar}")
    print(title.center(total_w))
    print(bar)

    row_fmt = "| " + " | ".join(f"{{:>{w}}}" for w in col_w) + " |"
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"

    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for m in result.worker_metrics:
        load = m.latency_ns / result.makespan * 100 if result.makespan else 0.0
        print(row_fmt.format(
            m.worker_id,
            m.iterations,
            m.cache_misses,
            m.latency_ns,
            m.slack_ns,
            f"{load:.1f}%",
        ))

    print(sep)

    avg_latency = sum(m.latency_ns for m in result.worker_metrics) / len(result.worker_metrics)
    efficiency = avg_latency / result.makespan * 100 if result.makespan else 0.0
    total_iters = sum(m.iterations for m in result.worker_metrics)
    total_misses = sum(m.cache_misses for m in result.worker_metrics)

    print(f"  Makespan      : {result.makespan:,} ns")
    print(f"  Total iters   : {total_iters:,}")
    print(f"  Total misses  : {total_misses:,}")
    print(f"  Efficiency    : {efficiency:.1f}%  (avg worker load / makespan)")
    print(bar)


simulator = Simulator(
    page_size=256,
    SRAM_size=4,
    iteration_cost=10,
    cache_miss_penalty=100,
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
dp = DataProcessor(tokenizer, "Alignment-Lab-AI/CodeInterpreterData-sharegpt")
out = dp.process(0)

NUM_WORKERS = 5
NUM_SPLITS = 4
TILE_SIZE = 2

print_result(
    f"FlashAttention  |  workers={NUM_WORKERS}",
    simulator.evaluate(flash_attention(out, NUM_WORKERS)),
)

print_result(
    f"FlashDecoding   |  workers={NUM_WORKERS}  splits={NUM_SPLITS}",
    simulator.evaluate(flash_decoding(out, NUM_WORKERS, NUM_SPLITS)),
)

print_result(
    f"LeanAttention   |  workers={NUM_WORKERS}  tile={TILE_SIZE}",
    simulator.evaluate(lean_attention(out, NUM_WORKERS, TILE_SIZE)),
)

print_result(
    f"FlashInfer   |  workers={NUM_WORKERS}  tile={TILE_SIZE}",
    simulator.evaluate(flashinfer(out, NUM_WORKERS, TILE_SIZE)),
)

print_result(
    f"FlashInfer-KK  |  workers={NUM_WORKERS}",
    simulator.evaluate(flashinfer_kk(out, NUM_WORKERS)),
)