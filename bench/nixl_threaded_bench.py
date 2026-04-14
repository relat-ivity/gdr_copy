#!/usr/bin/env python3

"""
Python NIXL benchmark with configurable multi-threaded submission.

Usage:
  python bench/nixl_threaded_bench.py [gpu_id] [nic_name]
  python bench/nixl_threaded_bench.py 0 mlx5_0 --submit-threads 2

This benchmark mirrors the NIXL line in bench/bench.cpp:
  - issue latency: only the postXferReq call
  - bandwidth: total bytes / total time for a post-all + wait-all batch

The timing model intentionally matches the current C++ bench:
  - single process
  - one NIXL initiator + target loopback pair per submit thread
  - warmup runs are sequential post+wait and are not measured
  - measured runs pre-create handles, then post all, then wait all
  - each submit thread owns its own full batch and waits for its own batch
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Sequence


def _fail(msg: str) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(2)


if not sys.platform.startswith("linux"):
    _fail("This script only supports Linux because NIXL UCX/GDR is Linux-only.")

try:
    import torch
except ModuleNotFoundError as exc:
    _fail(f"PyTorch is required: {exc}")

try:
    import nixl._bindings as nixl
except ModuleNotFoundError as exc:
    _fail(f"NIXL Python bindings are required: {exc}")


def now_us() -> float:
    return time.perf_counter_ns() / 1e3


@dataclass
class BenchResult:
    median_us: float = 0.0
    p99_us: float = 0.0
    bw_GBs: float = 0.0


@dataclass
class BenchPair:
    issue: BenchResult
    transfer: BenchResult


@dataclass
class DirectionRow:
    bytes: int
    nixl: BenchPair


class BufferBundle:
    def __init__(
        self,
        host_tensor,
        device_tensor,
        op,
        initiator_regs,
        target_regs,
        local_descs,
        remote_descs,
    ) -> None:
        self.host_tensor = host_tensor
        self.device_tensor = device_tensor
        self.op = op
        self.initiator_regs = initiator_regs
        self.target_regs = target_regs
        self.local_descs = local_descs
        self.remote_descs = remote_descs


def analyse(samples: List[float], bytes_count: int) -> BenchResult:
    if not samples:
        return BenchResult()

    ordered = sorted(samples)
    n = len(ordered)
    median = ordered[n // 2]
    p99 = ordered[min(n - 1, int(n * 0.99))]
    avg = sum(ordered) / n
    bw = (bytes_count / 1e9) / (avg / 1e6) if avg > 0.0 else 0.0
    return BenchResult(median_us=median, p99_us=p99, bw_GBs=bw)


def format_size(bytes_count: int) -> str:
    if bytes_count < (1 << 10):
        return f"{bytes_count}B"
    if bytes_count < (1 << 20):
        return f"{bytes_count >> 10}KiB"
    return f"{bytes_count >> 20}MiB"


def print_table(title: str, rows: Sequence[DirectionRow], issue_table: bool) -> None:
    print(f"\n--- {title} ---")
    if issue_table:
        print(f"{'Size':<12} | {'NIXL (median / p99)':<23}")
        print(f"{'-' * 12}-+-{'-' * 23}")
        for row in rows:
            res = row.nixl.issue
            print(
                f"{format_size(row.bytes):<12} | "
                f"{res.median_us:7.2f} us / {res.p99_us:7.2f} us"
            )
    else:
        print(f"{'Size':<12} | {'NIXL (BW)':<16}")
        print(f"{'-' * 12}-+-{'-' * 16}")
        for row in rows:
            res = row.nixl.transfer
            print(f"{format_size(row.bytes):<12} | {res.bw_GBs:8.2f} GB/s")


class NixlLoopbackPair:
    def __init__(self, gpu_id: int, nic_name: str, name_suffix: str = "") -> None:
        self.initiator_name = f"py-bench-init-{gpu_id}{name_suffix}"
        self.target_name = f"py-bench-target-{gpu_id}{name_suffix}"
        self.initiator = nixl.nixlAgent(self.initiator_name, self._make_agent_config())
        self.target = nixl.nixlAgent(self.target_name, self._make_agent_config())

        initiator_params, _ = self.initiator.getPluginParams("UCX")
        target_params, _ = self.target.getPluginParams("UCX")
        self._configure_ucx_params(initiator_params, nic_name)
        self._configure_ucx_params(target_params, nic_name)

        self.initiator_backend = self.initiator.createBackend("UCX", initiator_params)
        self.target_backend = self.target.createBackend("UCX", target_params)
        self.remote_md_loaded = False

    @staticmethod
    def _make_agent_config():
        cfg = nixl.nixlAgentConfig()
        cfg.useProgThread = True
        cfg.useListenThread = False
        if hasattr(nixl, "NIXL_THREAD_SYNC_DEFAULT"):
            cfg.syncMode = nixl.NIXL_THREAD_SYNC_DEFAULT
        cfg.pthrDelay = 0
        return cfg

    @staticmethod
    def _configure_ucx_params(params: dict, nic_name: str) -> None:
        params["device_list"] = nic_name
        params["ucx_devices"] = nic_name

    def register_buffers(self, initiator_regs, target_regs) -> None:
        self.initiator.registerMem(initiator_regs, [self.initiator_backend])
        self.target.registerMem(target_regs, [self.target_backend])
        target_md = self.target.getLocalMD()
        self.initiator.loadRemoteMD(target_md)
        self.remote_md_loaded = True

    def deregister_buffers(self, initiator_regs, target_regs) -> None:
        if self.remote_md_loaded:
            self.initiator.invalidateRemoteMD(self.target_name)
            self.remote_md_loaded = False
        self.initiator.deregisterMem(initiator_regs, [self.initiator_backend])
        self.target.deregisterMem(target_regs, [self.target_backend])

    def create_request(self, op, local_descs, remote_descs):
        return self.initiator.createXferReq(
            op,
            local_descs,
            remote_descs,
            self.target_name,
            "",
            [self.initiator_backend],
        )

    def post_request(self, handle):
        return self.initiator.postXferReq(handle)

    def wait_request(self, handle):
        while True:
            status = self.initiator.getXferStatus(handle)
            if status == nixl.NIXL_IN_PROG:
                time.sleep(0)
                continue
            return status

    def release_request(self, handle) -> None:
        self.initiator.releaseXferReq(handle)


def make_buffers(bytes_count: int, direction: str, gpu_id: int) -> BufferBundle:
    torch.cuda.set_device(gpu_id)

    host_tensor = torch.empty(bytes_count, dtype=torch.uint8, pin_memory=True)
    device_tensor = torch.empty(bytes_count, dtype=torch.uint8, device=f"cuda:{gpu_id}")

    if direction == "h2d":
        host_tensor.fill_(0xA5)
        device_tensor.zero_()
        op = nixl.NIXL_WRITE
    elif direction == "d2h":
        host_tensor.zero_()
        device_tensor.fill_(0x5A)
        op = nixl.NIXL_READ
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    torch.cuda.synchronize(gpu_id)

    host_ptr = int(host_tensor.data_ptr())
    device_ptr = int(device_tensor.data_ptr())

    initiator_regs = nixl.nixlRegDList(nixl.DRAM_SEG)
    initiator_regs.addDesc((host_ptr, bytes_count, 0, "host"))

    target_regs = nixl.nixlRegDList(nixl.VRAM_SEG)
    target_regs.addDesc((device_ptr, bytes_count, gpu_id, "gpu"))

    local_descs = nixl.nixlXferDList(nixl.DRAM_SEG)
    local_descs.addDesc((host_ptr, bytes_count, 0))

    remote_descs = nixl.nixlXferDList(nixl.VRAM_SEG)
    remote_descs.addDesc((device_ptr, bytes_count, gpu_id))

    return BufferBundle(
        host_tensor=host_tensor,
        device_tensor=device_tensor,
        op=op,
        initiator_regs=initiator_regs,
        target_regs=target_regs,
        local_descs=local_descs,
        remote_descs=remote_descs,
    )


def run_nixl_timings(
    bytes_count: int,
    direction: str,
    warmup: int,
    iters: int,
    gpu_id: int,
    nic_name: str,
    submit_threads: int,
) -> BenchPair:
    if submit_threads < 1:
        submit_threads = 1

    issue_samples_by_thread: List[List[float]] = [[] for _ in range(submit_threads)]
    ready_barrier = threading.Barrier(submit_threads + 1)
    start_event = threading.Event()
    done_event = threading.Event()
    done_lock = threading.Lock()
    error_lock = threading.Lock()

    done_count = 0
    errors: List[str] = []

    def record_error(message: str) -> None:
        with error_lock:
            if not errors:
                errors.append(message)
        done_event.set()
        start_event.set()
        try:
            ready_barrier.abort()
        except threading.BrokenBarrierError:
            pass

    def worker(tid: int) -> None:
        nonlocal done_count
        handles = []
        pair = None
        buffers = None

        try:
            torch.cuda.set_device(gpu_id)
            pair = NixlLoopbackPair(
                gpu_id=gpu_id,
                nic_name=nic_name,
                name_suffix=f"-t{tid}-{direction}-{bytes_count}",
            )
            buffers = make_buffers(bytes_count, direction, gpu_id)
            pair.register_buffers(buffers.initiator_regs, buffers.target_regs)

            handle_count = max(warmup, iters)
            for _ in range(handle_count):
                handles.append(
                    pair.create_request(
                        buffers.op,
                        buffers.local_descs,
                        buffers.remote_descs,
                    )
                )

            for i in range(warmup):
                status = pair.post_request(handles[i])
                if status not in (nixl.NIXL_SUCCESS, nixl.NIXL_IN_PROG):
                    record_error(
                        f"[nixl] warmup post failed: direction={direction} "
                        f"tid={tid} iter={i} rc={int(status)}"
                    )
                    return
                status = pair.wait_request(handles[i])
                if status != nixl.NIXL_SUCCESS:
                    record_error(
                        f"[nixl] warmup wait failed: direction={direction} "
                        f"tid={tid} iter={i} rc={int(status)}"
                    )
                    return

            ready_barrier.wait()
            start_event.wait()

            thread_samples = issue_samples_by_thread[tid]
            for i in range(iters):
                ti = now_us()
                status = pair.post_request(handles[i])
                thread_samples.append(now_us() - ti)
                if status not in (nixl.NIXL_SUCCESS, nixl.NIXL_IN_PROG):
                    record_error(
                        f"[nixl] measured post failed: direction={direction} "
                        f"tid={tid} iter={i} rc={int(status)}"
                    )
                    return

            for i in range(iters):
                status = pair.wait_request(handles[i])
                if status != nixl.NIXL_SUCCESS:
                    record_error(
                        f"[nixl] measured wait failed: direction={direction} "
                        f"tid={tid} iter={i} rc={int(status)}"
                    )
                    return

            with done_lock:
                done_count += 1
                if done_count == submit_threads:
                    done_event.set()

        except threading.BrokenBarrierError:
            if not errors:
                record_error(
                    f"[nixl] start barrier broken before measurement: "
                    f"direction={direction} tid={tid}"
                )
        except Exception as exc:  # pragma: no cover - defensive path
            record_error(f"[nixl] worker raised: direction={direction} tid={tid} err={exc}")
        finally:
            if pair is not None:
                for handle in handles:
                    try:
                        pair.release_request(handle)
                    except Exception:
                        pass
            if pair is not None and buffers is not None:
                try:
                    pair.deregister_buffers(buffers.initiator_regs, buffers.target_regs)
                except Exception:
                    pass
            del buffers
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(gpu_id)
                except Exception:
                    pass

    workers = [
        threading.Thread(target=worker, args=(tid,), name=f"nixl-submit-{tid}")
        for tid in range(submit_threads)
    ]
    for worker_thread in workers:
        worker_thread.start()

    try:
        ready_barrier.wait()
    except threading.BrokenBarrierError:
        for worker_thread in workers:
            worker_thread.join()
        _fail(errors[0] if errors else "[nixl] worker setup failed before measurement")

    t0 = now_us()
    start_event.set()
    done_event.wait()
    total_us = now_us() - t0

    for worker_thread in workers:
        worker_thread.join()

    if errors:
        _fail(errors[0])

    issue_samples: List[float] = []
    for per_thread_samples in issue_samples_by_thread:
        issue_samples.extend(per_thread_samples)

    out = BenchPair(issue=analyse(issue_samples, bytes_count), transfer=BenchResult())
    out.transfer.median_us = total_us
    out.transfer.p99_us = total_us
    out.transfer.bw_GBs = (
        (
            float(bytes_count)
            * float(iters)
            * float(submit_threads)
            / 1e9
        )
        / (total_us / 1e6)
        if iters > 0 and total_us > 0.0
        else 0.0
    )
    return out


def configure_ucx_env() -> None:
    os.environ.setdefault("UCX_TLS", "rc_x,cuda_copy,cuda_ipc")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "yes")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_ZCOPY_THRESH", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python multi-threaded NIXL GDR benchmark")
    parser.add_argument("gpu_id", nargs="?", type=int, default=0)
    parser.add_argument("nic_name", nargs="?", default="mlx5_0")
    parser.add_argument("--submit-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_ucx_env()

    torch.cuda.set_device(args.gpu_id)
    prop = torch.cuda.get_device_properties(args.gpu_id)

    print("=================================================================")
    print(f"  NIXL Python Threaded Benchmark  -  GPU {args.gpu_id}  NIC {args.nic_name}")
    print("=================================================================\n")
    print(f"GPU: {prop.name}\n")
    print("Transport: NIXL-UCX GPUDirect RDMA")
    print(
        "Benchmark mode: issue latency + per-thread wait-all bandwidth "
        f"(threads={max(1, args.submit_threads)})\n"
    )

    sizes = []
    size = 4096
    while size <= (64 << 20):
        sizes.append(size)
        size *= 4

    h2d_rows: List[DirectionRow] = []
    d2h_rows: List[DirectionRow] = []

    for bytes_count in sizes:
        pair = run_nixl_timings(
            bytes_count=bytes_count,
            direction="h2d",
            warmup=args.warmup,
            iters=args.iters,
            gpu_id=args.gpu_id,
            nic_name=args.nic_name,
            submit_threads=args.submit_threads,
        )
        h2d_rows.append(DirectionRow(bytes=bytes_count, nixl=pair))

    for bytes_count in sizes:
        pair = run_nixl_timings(
            bytes_count=bytes_count,
            direction="d2h",
            warmup=args.warmup,
            iters=args.iters,
            gpu_id=args.gpu_id,
            nic_name=args.nic_name,
            submit_threads=args.submit_threads,
        )
        d2h_rows.append(DirectionRow(bytes=bytes_count, nixl=pair))

    print_table("Host->Device Issue Latency", h2d_rows, True)
    print_table("Host->Device Bandwidth", h2d_rows, False)
    print_table("Device->Host Issue Latency", d2h_rows, True)
    print_table("Device->Host Bandwidth", d2h_rows, False)

    total_measured_ops = len(sizes) * 2 * args.iters * max(1, args.submit_threads)
    total_measured_bytes = sum(sizes) * 2 * args.iters * max(1, args.submit_threads)
    print("\n=================================================================")
    print(f"Total measured ops: {total_measured_ops}")
    print(f"Total measured bytes: {total_measured_bytes / float(1 << 30):.2f} GiB")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
