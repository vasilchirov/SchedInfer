from heapq import heapify, heappop, heappush
from math import ceil
from models import Request, WorkloadChunk

def _kv_offsets(requests: list[Request]) -> list[int]:
    offsets, pos = [], 0
    for req in requests:
        offsets.append(pos)
        pos += req.l_kv
    return offsets

def flashinfer(
    requests: list[Request],
    num_workers: int,
    query_tile_size: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[list[WorkloadChunk]]:
    """
    FlashInfer balanced scheduling (Fully Corrected to match Algorithm 1).
    """
    kv_offsets = _kv_offsets(requests)

    # Step 1 & 2: Calculate total work and maximum chunk size L_kv
    # Matches Equation (1) and Line 3 from the paper diagram
    total_work = sum(
        ceil(req.l_qo / query_tile_size) * req.l_kv
        for req in requests
    )
    max_chunk_size = max(1, ceil(total_work / num_workers))

    chunks: list[WorkloadChunk] = []

    # Step 3: Split along BOTH Query tiles and KV sequences (Line 4 of the paper)
    for req, kv_base in zip(requests, kv_offsets):
        num_q_tiles = ceil(req.l_qo / query_tile_size)
        
        # For each independent query tile of this request
        for q_tile in range(num_q_tiles):
            # Calculate how many tokens are in this specific query tile
            current_l_qo = min(query_tile_size, req.l_qo - (q_tile * query_tile_size))
            
            # Slice its associated KV sequence into pieces no larger than L_kv
            for offset in range(0, req.l_kv, max_chunk_size):
                l_kv = min(max_chunk_size, req.l_kv - offset)

                chunks.append(
                    WorkloadChunk(
                        req_id=req.id,
                        l_qo=current_l_qo,  # Pass the split query tile size
                        l_kv=l_kv,
                        kv_start=kv_base + offset,
                    )
                )

    # Step 4: Sort chunks in descending order of KV length (Line 5)
    chunks.sort(key=lambda c: c.l_kv, reverse=True)

    queues: list[list[WorkloadChunk]] = [[] for _ in range(num_workers)]

    # Step 5: Initialize Min-Priority Queue for CTA loads (Line 6)
    worker_heap = [(0.0, worker_id) for worker_id in range(num_workers)]
    heapify(worker_heap)
    
    # Step 6: Greedy Assignment using the paper's cost function (Lines 7-13)
    for chunk in chunks:
        current_cost, worker_id = heappop(worker_heap)
        queues[worker_id].append(chunk)

        # The paper uses the query tile scale factor T_q (or chunk.l_qo here)
        # to charge the worker load
        chunk_cost = (
            alpha * chunk.l_qo +
            beta * chunk.l_kv
        )

        heappush(
            worker_heap,
            (current_cost + chunk_cost, worker_id),
        )

    return queues