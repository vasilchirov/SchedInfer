from simulator import Simulator
from data_processing import load_requests
from schedules import Scheduler, Strategy
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
        load = (
            m.latency_ns / result.makespan * 100
            if result.makespan else 0.0
        )

        print(row_fmt.format(
            m.worker_id,
            m.iterations,
            m.cache_misses,
            m.latency_ns,
            m.slack_ns,
            f"{load:.1f}%"
        ))

    print(sep)

    total_iters = sum(m.iterations for m in result.worker_metrics)
    total_misses = sum(m.cache_misses for m in result.worker_metrics)

    avg_latency = (
        sum(m.latency_ns for m in result.worker_metrics)
        / len(result.worker_metrics)
    )

    efficiency = (
        avg_latency / result.makespan * 100
        if result.makespan else 0.0
    )

    print(f"  Makespan          : {result.makespan:,} ns")
    print(f"  Decode steps      : {result.num_steps:,}")
    print(f"  Avg step latency  : {result.avg_step_latency:,.2f} ns")
    print(f"  Throughput        : {result.throughput:,.2f} tokens/s")
    print(f"  Total iterations  : {total_iters:,}")
    print(f"  Total cache misses: {total_misses:,}")
    print(f"  Efficiency        : {efficiency:.1f}%")
    print(bar)

scheduler = Scheduler()

simulator = Simulator(
    scheduler=scheduler,
    num_workers=5,
    iteration_cost=10,
    cache_miss_penalty=100,
    page_size=32,
    sram_size=32
)

out = load_requests(0)

print_result(
    f"FlashAttention  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_ATTENTION, requests=out),
)

print_result(
    f"FlashAttention  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_ATTENTION, requests=out, enable_cache=True),
)

print_result(
    f"FlashDecoding  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_DECODING, requests=out),
)

print_result(
    f"FlashDecoding  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_DECODING, requests=out, enable_cache=True),
)


print_result(
    f"LeanAttention  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.LEAN_ATTENTION, requests=out),
)

print_result(
    f"LeanAttention  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.LEAN_ATTENTION, requests=out, enable_cache=True),
)

print_result(
    f"FlashInfer  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_INFER, requests=out),
)

print_result(
    f"FlashInfer  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_INFER, requests=out, enable_cache=True),
)

print_result(
    f"FlashInfer-kk  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_INFER_KK, requests=out),
)

print_result(
    f"FlashInfer-kk  |  workers={scheduler.num_workers}",
    simulator.simulate(sched_strategy=Strategy.FLASH_INFER_KK, requests=out, enable_cache=True),
)
