from math import ceil

from models import WorkloadChunk, WorkerMetrics, EvaluationResult

class Simulator:
    def __init__(self, page_size: int, SRAM_size: int, iteration_cost: int, cache_miss_penalty: int):
        self.page_size = page_size
        self.SRAM_size = SRAM_size
        self.iteration_cost = iteration_cost
        self.cache_miss_penalty = cache_miss_penalty

    def evaluate(self, cta_queues: list[list[WorkloadChunk]]):
        worker_metrics_list = []
        num_workers = len(cta_queues)
        makespan = 0

        for i in range(num_workers):
            chunks = cta_queues[i]
            cache_misses = 0
            iterations = 0

            SRAM = []
            for chunk in chunks:
                first_page = chunk.kv_start // self.page_size
                last_page = (chunk.kv_start + chunk.l_kv - 1) // self.page_size
                
                for p in range(first_page, last_page + 1):
                    if p not in SRAM:
                        cache_misses += 1
                        if len(SRAM) >= self.SRAM_size:
                            SRAM.pop(0)
                        
                        SRAM.append(p)

                chunk_iterations = ceil((chunk.l_qo * chunk.l_kv) / 32)
                iterations += chunk_iterations
            
            latency_ns = iterations * self.iteration_cost + cache_misses * self.cache_miss_penalty
            makespan = max(makespan, latency_ns)

            worker_metrics_list.append(WorkerMetrics(
                worker_id = i,
                iterations = iterations,
                cache_misses = cache_misses,
                latency_ns = latency_ns,
                slack_ns = 0
            ))

        for i in range(len(worker_metrics_list)):
            worker_metrics_list[i].slack_ns = makespan - worker_metrics_list[i].latency_ns
        
        return EvaluationResult(
            makespan=makespan,
            worker_metrics=worker_metrics_list
        )

if __name__ == "__main__":
    simulator = Simulator(
        page_size=256,
        SRAM_size=4,
        iteration_cost=10,
        cache_miss_penalty=100
    )

    cta_queues = [
        [
            WorkloadChunk(
                req_id=1,
                l_qo=1,
                l_kv=1000,
                kv_start=0
            )
        ],

        [
            WorkloadChunk(
                req_id=2,
                l_qo=1,
                l_kv=500,
                kv_start=1000
            ),
            WorkloadChunk(
                req_id=3,
                l_qo=1,
                l_kv=250,
                kv_start=1500
            )
        ],

        [
            WorkloadChunk(
                req_id=4,
                l_qo=1,
                l_kv=2000,
                kv_start=2000
            )
        ]
    ]

    result = simulator.evaluate(cta_queues)

    print(f"Makespan: {result.makespan} ns\n")

    for metrics in result.worker_metrics:
        print(
            f"Worker {metrics.worker_id}: "
            f"iterations={metrics.iterations}, "
            f"cache_misses={metrics.cache_misses}, "
            f"latency={metrics.latency_ns} ns, "
            f"slack={metrics.slack_ns} ns"
        )
