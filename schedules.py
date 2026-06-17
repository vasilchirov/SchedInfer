from enum import Enum
from heapq import heapify, heappop, heappush
from math import ceil

from models import Request, WorkloadChunk

def _kv_offsets(requests: list[Request]) -> list[int]:
    offsets, pos = [], 0
    for req in requests:
        offsets.append(pos)
        pos += req.l_kv
    return offsets


def flash_attention(requests: list[Request], num_workers: int) -> list[list[WorkloadChunk]]:
    """
    FlashAttention: one chunk per request, round-robin across workers.
    No KV splitting — each assigned worker iterates the full KV range of its requests.
    """
    queues: list[list[WorkloadChunk]] = [[] for _ in range(num_workers)]
    for i, (req, kv_base) in enumerate(zip(requests, _kv_offsets(requests))):
        queues[i % num_workers].append(
            WorkloadChunk(req_id=req.id, l_qo=req.l_qo, l_kv=req.l_kv, kv_start=kv_base)
        )
    return queues


def flash_decoding(requests: list[Request], num_workers: int, num_splits: int) -> list[list[WorkloadChunk]]:
    """
    FlashDecoding: each request's KV sequence is split into `num_splits` equal partitions.
    Partitions are distributed round-robin across workers; each worker later reduces
    its partial softmax outputs.
    """
    queues: list[list[WorkloadChunk]] = [[] for _ in range(num_workers)]
    chunk_idx = 0
    for req, kv_base in zip(requests, _kv_offsets(requests)):
        split_size = ceil(req.l_kv / num_splits)
        for s in range(num_splits):
            offset = s * split_size
            if offset >= req.l_kv:
                break
            l_kv = min(split_size, req.l_kv - offset)
            queues[chunk_idx % num_workers].append(
                WorkloadChunk(req_id=req.id, l_qo=req.l_qo, l_kv=l_kv, kv_start=kv_base + offset)
            )
            chunk_idx += 1
    return queues


def lean_attention(requests: list[Request], num_workers: int, tile_size: int) -> list[list[WorkloadChunk]]:
    """
    LeanAttention (stream-K): all KV tiles across all requests are linearised into a
    single sequence and then distributed as evenly as possible across workers.
    Consecutive tiles belonging to the same request are merged back into one chunk.
    """
    tiles: list[tuple[int, int, int, int]] = []  # (req_id, l_qo, global_kv_start, tile_l_kv)
    for req, kv_base in zip(requests, _kv_offsets(requests)):
        num_tiles = ceil(req.l_kv / tile_size)
        for t in range(num_tiles):
            offset = t * tile_size
            tiles.append((req.id, req.l_qo, kv_base + offset, min(tile_size, req.l_kv - offset)))

    total = len(tiles)
    base, remainder = divmod(total, num_workers)

    queues: list[list[WorkloadChunk]] = [[] for _ in range(num_workers)]
    tile_idx = 0
    for w in range(num_workers):
        count = base + (1 if w < remainder else 0)
        worker_tiles = tiles[tile_idx: tile_idx + count]
        tile_idx += count

        if not worker_tiles:
            continue

        req_id, l_qo, kv_start, l_kv = worker_tiles[0]
        for nreq_id, nl_qo, nkv_start, nl_kv in worker_tiles[1:]:
            if nreq_id == req_id and nkv_start == kv_start + l_kv:
                l_kv += nl_kv
            else:
                queues[w].append(WorkloadChunk(req_id=req_id, l_qo=l_qo, l_kv=l_kv, kv_start=kv_start))
                req_id, l_qo, kv_start, l_kv = nreq_id, nl_qo, nkv_start, nl_kv
        queues[w].append(WorkloadChunk(req_id=req_id, l_qo=l_qo, l_kv=l_kv, kv_start=kv_start))

    return queues


# ── FlashInfer (existing) ────────────────────────────────────────────────────

from heapq import heapify, heappop, heappush
from math import ceil

from models import Request, WorkloadChunk


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

# ── Karmarkar-Karp (new) ─────────────────────────────────────────────────────

def flashinfer_kk(
    requests: list[Request],
    num_workers: int,
) -> list[list[WorkloadChunk]]:
    """
    FlashInfer scheduling via k-way Karmarkar-Karp differencing.

    Unlike LPT (flashinfer above), KK repeatedly merges the two most-imbalanced
    partial assignments by pairing each heavy bin with the other's light bin.
    This global view produces tighter partitions when chunk costs vary widely.

    Cost model: ceil(l_qo * l_kv / 32), matching the simulator.
    """
    from itertools import count as icount

    kv_offsets = _kv_offsets(requests)

    total_work = sum(ceil(req.l_qo * req.l_kv / 32) for req in requests)
    target_work = max(1, ceil(total_work / num_workers))

    items: list[tuple[int, WorkloadChunk]] = []
    for req, kv_base in zip(requests, kv_offsets):
        kv_per_chunk = max(1, ceil(target_work * 32 / req.l_qo))
        for offset in range(0, req.l_kv, kv_per_chunk):
            l_kv = min(kv_per_chunk, req.l_kv - offset)
            cost = ceil(req.l_qo * l_kv / 32)
            items.append((cost, WorkloadChunk(
                req_id=req.id, l_qo=req.l_qo,
                l_kv=l_kv, kv_start=kv_base + offset,
            )))

    if not items:
        return [[] for _ in range(num_workers)]

    K = num_workers
    tiebreak = icount()

    # Heap entries: (-spread, unique_id, bin_loads_desc, assignments_parallel)
    # unique_id prevents Python from comparing the lists on ties.
    heap: list = []
    for cost, chunk in items:
        bins = [cost] + [0] * (K - 1)
        asgn = [[chunk]] + [[] for _ in range(K - 1)]
        heappush(heap, (-cost, next(tiebreak), bins, asgn))

    while len(heap) > 1:
        _, _, bins1, asgn1 = heappop(heap)
        _, _, bins2, asgn2 = heappop(heap)

        # Pair bins1[i] (heavy→light) with bins2[K-1-i] (light→heavy)
        merged = sorted(
            [(bins1[i] + bins2[K - 1 - i], asgn1[i] + asgn2[K - 1 - i])
             for i in range(K)],
            key=lambda x: -x[0],
        )
        new_bins = [m[0] for m in merged]
        new_asgn = [m[1] for m in merged]

        spread = new_bins[0] - new_bins[-1]
        heappush(heap, (-spread, next(tiebreak), new_bins, new_asgn))

    _, _, _, final_asgn = heap[0]
    return [final_asgn[w] for w in range(K)]

class Strategy(Enum):
    FLASH_ATTENTION = 1
    FLASH_DECODING = 2
    LEAN_ATTENTION = 3
    FLASH_INFER = 4
    FLASH_INFER_KK = 5

class Scheduler:
    def __init__(self, num_workers: int = 5, num_splits: int = 5, tile_size: int = 32, query_tile_size: int = 32):
        
        self.num_workers = num_workers
        self.num_splits = num_workers
        self.tile_size = tile_size
        self.query_tile_size = query_tile_size

    def run(self, sched_strategy: Strategy, requests: list[Request]) -> list[list[WorkloadChunk]]:
        if sched_strategy is Strategy.FLASH_ATTENTION:
            return flash_attention(requests, self.num_workers)
        if sched_strategy is Strategy.FLASH_DECODING:
            return flash_decoding(requests, self.num_workers, self.num_splits)
        if sched_strategy is Strategy.LEAN_ATTENTION:
            return lean_attention(requests, self.num_workers, self.tile_size)
        if sched_strategy is Strategy.FLASH_INFER:
            return flashinfer(requests, self.num_workers, self.query_tile_size)
        if sched_strategy is Strategy.FLASH_INFER_KK:
            return flashinfer_kk(requests, self.num_workers)

        raise ValueError(f"Unknown scheduling strategy: {sched_strategy}")
