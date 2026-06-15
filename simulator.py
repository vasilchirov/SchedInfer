from math import ceil
from copy import deepcopy

from models import WorkloadChunk, WorkerMetrics, EvaluationResult, Request
from schedules import Scheduler, Strategy

class Simulator:
    def __init__(self, num_workers: int, iteration_cost: int, scheduler: Scheduler):
        self.num_workers = num_workers
        self.eval = Evaluator(
            iteration_cost=iteration_cost
        )
        self.scheduler = scheduler

    def simulate(self, sched_strategy: Strategy, requests: list[Request]) -> EvaluationResult:
        requests = deepcopy(requests)

        # prefil phase
        schedule = self.scheduler.run(sched_strategy=sched_strategy, requests=requests)
        total_eval_result = self.eval.evaluate(schedule)

        # decode phase
        for r in requests:
            r.l_qo = 1
        
        generated_tokens = 0
        stop = False
        while not stop:
            schedule = self.scheduler.run(sched_strategy=sched_strategy, requests=requests)

            eval_result = self.eval.evaluate(schedule)

            for i in range(len(total_eval_result.worker_metrics)):
                    total_eval_result.worker_metrics[i].iterations += eval_result.worker_metrics[i].iterations
                    total_eval_result.worker_metrics[i].latency_ns += eval_result.worker_metrics[i].latency_ns
                    total_eval_result.worker_metrics[i].slack_ns += eval_result.worker_metrics[i].slack_ns
            
            total_eval_result.makespan += eval_result.makespan
            total_eval_result.num_steps += 1
            
            stop = True
            new_requests = []
            for r in requests:
                r.response_len -= 1
                r.l_kv += 1
                if r.response_len > 0:
                    stop = False
                    new_requests.append(r)

            generated_tokens += len(requests)
            requests = new_requests
        
        total_eval_result.avg_step_latency = total_eval_result.makespan / total_eval_result.num_steps
        total_eval_result.throughput = generated_tokens / total_eval_result.makespan
        return total_eval_result
    

class Evaluator:
    def __init__(self, iteration_cost: int):
        self.iteration_cost = iteration_cost

    def evaluate(self, cta_queues: list[list[WorkloadChunk]]) -> EvaluationResult:
        worker_metrics_list = []
        num_workers = len(cta_queues)
        makespan = 0

        for i in range(num_workers):
            chunks = cta_queues[i]
            iterations = 0

            for chunk in chunks:
                chunk_iterations = ceil((chunk.l_qo * chunk.l_kv) / 32)
                iterations += chunk_iterations
            
            latency_ns = iterations * self.iteration_cost
            makespan = max(makespan, latency_ns)

            worker_metrics_list.append(WorkerMetrics(
                worker_id = i,
                iterations = iterations,
                latency_ns = latency_ns,
                slack_ns = 0
            ))

        for i in range(len(worker_metrics_list)):
            worker_metrics_list[i].slack_ns = makespan - worker_metrics_list[i].latency_ns
        
        return EvaluationResult(
            makespan=makespan,
            num_steps=0,
            avg_step_latency=0,
            throughput=0.0,
            worker_metrics=worker_metrics_list
        )
