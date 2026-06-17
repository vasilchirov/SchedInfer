import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from data_processing import DataProcessor
from models import Request
from schedules import Scheduler, Strategy
from simulator import Simulator
from transformers import AutoTokenizer
DATA_DIR = Path("data_cache")
DATASET_NAME = "Alignment-Lab-AI/CodeInterpreterData-sharegpt"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"


def _strategy_label(strategy: Strategy) -> str:
    return strategy.name.replace("_", " ").title()


def _load_processor(max_rows: int | None = None) -> DataProcessor:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return DataProcessor(tokenizer, DATASET_NAME, max_rows=max_rows)


def _load_request_pool(num_rows: int = 200) -> list[Request]:
    processor = _load_processor(max_rows=num_rows)
    pool: list[Request] = []
    for row_id in range(len(processor.data_df)):
        pool.extend(processor.process(row_id))
    return pool


def _take_uniform(requests: list[Request], count: int) -> list[Request]:
    return sorted(requests, key=lambda r: r.l_kv)[:count]


def _take_skewed(requests: list[Request], count: int) -> list[Request]:
    ordered = sorted(requests, key=lambda r: r.l_kv)
    if count >= len(ordered):
        return ordered

    low_count = max(1, int(count * 0.8))
    high_count = count - low_count
    low_part = ordered[:low_count]
    high_part = ordered[-high_count:] if high_count > 0 else []
    return low_part + high_part


def _run_simulation(requests: list[Request], num_workers: int, strategy: Strategy) -> dict[str, object]:
    scheduler = Scheduler(num_workers=num_workers)
    simulator = Simulator(scheduler=scheduler, num_workers=num_workers)
    result = simulator.simulate(strategy, requests=requests, enable_cache=True)

    latencies = [m.latency_ns for m in result.worker_metrics]
    load_imbalance_ratio = max(latencies) / max(1, min(latencies))

    return {
        "makespan_ns": result.makespan,
        "num_steps": result.num_steps,
        "avg_step_latency_ns": result.avg_step_latency,
        "throughput_tokens_per_s": result.throughput,
        "lir": load_imbalance_ratio,
    }


def experiment_1_uniform_vs_skewed() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = _load_request_pool(num_rows=512)

    scenarios = {
        "uniform": _take_uniform(pool, 432),
        "skewed": _take_skewed(pool, 432),
    }
    strategies = [
        Strategy.FLASH_ATTENTION,
        Strategy.FLASH_DECODING,
        Strategy.LEAN_ATTENTION,
        Strategy.FLASH_INFER,
    ]
    worker_counts = [108]

    rows: list[dict[str, object]] = []
    for scenario_name, requests in scenarios.items():
        for num_workers in worker_counts:
            for strategy in strategies:
                metrics = _run_simulation(requests, num_workers, strategy)
                rows.append(
                    {
                        "experiment": "rq1_distribution",
                        "scenario": scenario_name,
                        "num_workers": num_workers,
                        "strategy": strategy.name,
                        "strategy_label": _strategy_label(strategy),
                        "avg_l_qo": sum(r.l_qo for r in requests) / len(requests),
                        "avg_l_kv": sum(r.l_kv for r in requests) / len(requests),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "rq1_uniform_vs_skewed.csv", index=False)
    _plot_distribution(df, DATA_DIR / "rq1_uniform_vs_skewed.png")
    return df


def experiment_2_worker_scaling() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = _load_request_pool(num_rows=512)
    requests = _take_skewed(pool, 432)
    strategies = [
        Strategy.FLASH_ATTENTION,
        Strategy.FLASH_DECODING,
        Strategy.LEAN_ATTENTION,
        Strategy.FLASH_INFER,
    ]
    worker_counts = [8, 32, 64, 108]

    rows: list[dict[str, object]] = []
    for num_workers in worker_counts:
        for strategy in strategies:
            metrics = _run_simulation(requests, num_workers, strategy)
            rows.append(
                {
                    "experiment": "rq1_worker_scaling",
                    "num_workers": num_workers,
                    "strategy": strategy.name,
                    "strategy_label": _strategy_label(strategy),
                    "avg_l_qo": sum(r.l_qo for r in requests) / len(requests),
                    "avg_l_kv": sum(r.l_kv for r in requests) / len(requests),
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "rq1_worker_scaling.csv", index=False)
    _plot_worker_scaling(df, DATA_DIR / "rq1_worker_scaling.png")
    return df


def experiment_3_solver_comparison() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = _load_request_pool(num_rows=512)
    requests = _take_skewed(pool, 432)
    strategies = [Strategy.FLASH_INFER, Strategy.FLASH_INFER_KK]

    rows: list[dict[str, object]] = []
    for strategy in strategies:
        metrics = _run_simulation(requests, 108, strategy)
        rows.append(
            {
                "experiment": "rq2_solver_comparison",
                "num_workers": 108,
                "strategy": strategy.name,
                "strategy_label": _strategy_label(strategy),
                "avg_l_qo": sum(r.l_qo for r in requests) / len(requests),
                "avg_l_kv": sum(r.l_kv for r in requests) / len(requests),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "rq2_solver_comparison.csv", index=False)
    _plot_solver_comparison(df, DATA_DIR / "rq2_solver_comparison.png")
    return df


def experiment_4_solver_scaling() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool = _load_request_pool(num_rows=512)
    requests = _take_skewed(pool, 432)
    strategies = [Strategy.FLASH_INFER, Strategy.FLASH_INFER_KK]
    worker_counts = [8, 32, 64, 108]

    rows: list[dict[str, object]] = []
    for num_workers in worker_counts:
        for strategy in strategies:
            metrics = _run_simulation(requests, num_workers, strategy)
            rows.append(
                {
                    "experiment": "rq2_solver_scaling",
                    "num_workers": num_workers,
                    "strategy": strategy.name,
                    "strategy_label": _strategy_label(strategy),
                    "avg_l_qo": sum(r.l_qo for r in requests) / len(requests),
                    "avg_l_kv": sum(r.l_kv for r in requests) / len(requests),
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "rq2_solver_scaling.csv", index=False)
    _plot_solver_scaling(df, DATA_DIR / "rq2_solver_scaling.png")
    return df


def _plot_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values(["strategy", "scenario"])
    strategies = plot_df["strategy"].unique().tolist()
    scenarios = ["uniform", "skewed"]
    x = range(len(strategies))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for idx, scenario in enumerate(scenarios):
        subset = plot_df[plot_df["scenario"] == scenario].set_index("strategy").reindex(strategies)
        offset = (-width / 2) if idx == 0 else (width / 2)
        positions = [p + offset for p in x]
        label = scenario.title()
        axes[0].bar(
            positions,
            subset["makespan_ns"].tolist(),
            width=width,
            label=label,
        )
        axes[1].bar(
            positions,
            subset["lir"].tolist(),
            width=width,
            label=label,
        )

    axes[0].set_title("RQ1: Makespan")
    axes[1].set_title("RQ1: Load Imbalance Ratio")
    axes[0].set_ylabel("Makespan (ns)")
    axes[1].set_ylabel("LIR")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([_strategy_label(Strategy[s]) for s in strategies], rotation=20)
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_worker_scaling(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values(["strategy", "num_workers"])
    strategies = plot_df["strategy"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for strategy in strategies:
        subset = plot_df[plot_df["strategy"] == strategy]
        label = subset["strategy_label"].iloc[0]
        axes[0].plot(subset["num_workers"], subset["makespan_ns"], marker="o", label=label)
        axes[1].plot(subset["num_workers"], subset["lir"], marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 64, 108])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_title("RQ1: Makespan vs Workers")
    axes[1].set_title("RQ1: LIR vs Workers")
    axes[0].set_xlabel("Workers")
    axes[1].set_xlabel("Workers")
    axes[0].set_ylabel("Makespan (ns)")
    axes[1].set_ylabel("LIR")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_solver_comparison(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for strategy in df["strategy"].unique():
        subset = df[df["strategy"] == strategy]
        label = subset["strategy_label"].iloc[0]
        axes[0].bar(label, subset["makespan_ns"].iloc[0])
        axes[1].bar(label, subset["lir"].iloc[0])
    axes[0].set_title("RQ2: Solver Makespan")
    axes[1].set_title("RQ2: Solver LIR")
    axes[0].set_ylabel("Makespan (ns)")
    axes[1].set_ylabel("LIR")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_solver_scaling(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values(["strategy", "num_workers"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for strategy in plot_df["strategy"].unique():
        subset = plot_df[plot_df["strategy"] == strategy]
        label = subset["strategy_label"].iloc[0]
        axes[0].plot(subset["num_workers"], subset["makespan_ns"], marker="o", label=label)
        axes[1].plot(subset["num_workers"], subset["lir"], marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 64, 108])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_title("RQ2: Solver Makespan vs Workers")
    axes[1].set_title("RQ2: Solver LIR vs Workers")
    axes[0].set_xlabel("Workers")
    axes[1].set_xlabel("Workers")
    axes[0].set_ylabel("Makespan (ns)")
    axes[1].set_ylabel("LIR")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    experiment_1_uniform_vs_skewed()
    experiment_2_worker_scaling()
    experiment_3_solver_comparison()
    experiment_4_solver_scaling()
