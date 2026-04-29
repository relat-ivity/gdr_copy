/**
 * bench_gdr_full_gpu_mr.cpp - GDR issue latency and bandwidth with one large GPU MR
 *
 * Usage:
 *   sudo ./build/bench_gdr_full_gpu_mr <gpu_id> <nic_name>
 *   sudo ./build/bench_gdr_full_gpu_mr 4 mlx5_4
 *
 * This benchmark allocates one large GPU buffer, pins the whole buffer as a
 * single GPU MR window, and then measures GDR issue latency and bandwidth for
 * addresses that stay inside that pre-registered region.
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

static constexpr int WARMUP = 100;
static constexpr int ITERS  = 1000;
static constexpr size_t MAX_SWEEP_BYTES = 64ULL << 20;
static constexpr size_t GPU_MR_RESERVE_BYTES = 512ULL << 20;
static constexpr size_t GPU_MR_STEP_BYTES = 256ULL << 20;

struct BenchResult {
    double median_us = 0.0;
    double p99_us = 0.0;
    double bw_gbs = 0.0;
};

struct BenchPair {
    BenchResult issue;
    BenchResult transfer;
};

struct DirectionRow {
    size_t bytes = 0;
    BenchPair gdr{};
};

static double now_us() {
    using namespace std::chrono;
    return duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

static void die_cuda(cudaError_t rc, const char* what, int gpu_id, size_t bytes) {
    if (rc == cudaSuccess)
        return;
    fprintf(stderr, "[full-gpu-mr] %s failed: gpu=%d bytes=%zu err=%s\n",
            what, gpu_id, bytes, cudaGetErrorString(rc));
    std::exit(2);
}

static void format_size(size_t bytes, char* out, size_t out_len) {
    if (bytes < (1 << 10))      snprintf(out, out_len, "%zuB", bytes);
    else if (bytes < (1 << 20)) snprintf(out, out_len, "%zuKiB", bytes >> 10);
    else                        snprintf(out, out_len, "%zuMiB", bytes >> 20);
}

static BenchResult analyse(std::vector<double>& samples, size_t bytes) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    BenchResult r{};
    r.median_us = samples[n / 2];
    r.p99_us = samples[std::min(n - 1, static_cast<size_t>(n * 0.99))];
    const double avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);
    r.bw_gbs = (bytes / 1e9) / (avg / 1e6);
    return r;
}

static void pin_windows_or_die(const std::shared_ptr<GDRCopyChannel>& ch,
                               void* host_ptr, size_t host_bytes,
                               void* gpu_ptr, size_t gpu_bytes,
                               const char* label)
{
    if (ch->pin_host_window(host_ptr, host_bytes) != 0) {
        fprintf(stderr, "[full-gpu-mr] pin_host_window failed: case=%s bytes=%zu\n",
                label, host_bytes);
        std::exit(2);
    }
    if (ch->pin_gpu_window(gpu_ptr, gpu_bytes) != 0) {
        fprintf(stderr, "[full-gpu-mr] pin_gpu_window failed: case=%s bytes=%zu\n",
                label, gpu_bytes);
        std::exit(2);
    }
}

static BenchPair run_gdr_timings(const std::shared_ptr<GDRCopyChannel>& ch,
                                 void* dst, const void* src,
                                 size_t bytes, GDRCopyKind kind,
                                 int warmup, int iters)
{
    ch->reset_stats();
    std::vector<double> issue_samples;
    issue_samples.reserve(iters);

    // Measure pure submit cost with an immediate drain after every request so
    // queueing and backpressure do not leak into the issue samples.
    for (int i = 0; i < warmup + iters; ++i) {
        const double t0 = now_us();
        int rc = ch->memcpy_async(dst, src, bytes, kind);
        const double t1 = now_us();
        if (rc != 0) {
            fprintf(stderr,
                    "[full-gpu-mr] issue failed: rc=%d kind=%d bytes=%zu iter=%d/%d\n",
                    rc, static_cast<int>(kind), bytes, i, warmup + iters);
            std::exit(2);
        }
        while (true) {
            int sc = ch->sync();
            if (sc == 0)
                break;
            if (sc == -EAGAIN)
                continue;
            fprintf(stderr,
                    "[full-gpu-mr] sync failed: rc=%d kind=%d bytes=%zu iter=%d/%d\n",
                    sc, static_cast<int>(kind), bytes, i, warmup + iters);
            std::exit(2);
        }
        if (i >= warmup)
            issue_samples.push_back(t1 - t0);
    }

    auto run_batch = [&](int count, bool measure) -> double {
        if (count <= 0)
            return 0.0;

        const double t0 = measure ? now_us() : 0.0;
        int issued = 0;
        int done = 0;

        while (issued < count) {
            int rc = ch->memcpy_async(dst, src, bytes, kind);
            if (rc == 0) {
                ++issued;
                continue;
            }
            if (rc == -EBUSY) {
                while (done < issued) {
                    int sc = ch->sync();
                    if (sc == 0) {
                        ++done;
                        break;
                    }
                    if (sc == -EAGAIN)
                        continue;
                    fprintf(stderr,
                            "[full-gpu-mr] sync failed: rc=%d kind=%d bytes=%zu done=%d/%d\n",
                            sc, static_cast<int>(kind), bytes, done, count);
                    std::exit(2);
                }
                continue;
            }
            fprintf(stderr,
                    "[full-gpu-mr] issue failed: rc=%d kind=%d bytes=%zu issued=%d/%d\n",
                    rc, static_cast<int>(kind), bytes, issued, count);
            std::exit(2);
        }

        while (done < count) {
            int sc = ch->sync();
            if (sc == 0) {
                ++done;
                continue;
            }
            if (sc == -EAGAIN)
                continue;
            fprintf(stderr,
                    "[full-gpu-mr] sync failed: rc=%d kind=%d bytes=%zu done=%d/%d\n",
                    sc, static_cast<int>(kind), bytes, done, count);
            std::exit(2);
        }

        return measure ? (now_us() - t0) : 0.0;
    };

    run_batch(warmup, false);
    const double total_us = run_batch(iters, true);

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer.median_us = total_us;
    out.transfer.p99_us = total_us;
    out.transfer.bw_gbs =
        (iters > 0 && total_us > 0.0)
            ? ((static_cast<double>(bytes) * static_cast<double>(iters)) / 1e9) / (total_us / 1e6)
            : 0.0;
    return out;
}

static void print_issue_table(const char* title, const std::vector<DirectionRow>& rows) {
    printf("\n--- %s ---\n", title);
    printf("%-12s | %-21s\n", "Size", "   GDR (median / p99)");
    printf("%-12s-+-%-21s\n", "------------", "-----------------------");

    for (const auto& row : rows) {
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));
        printf("%-12s | %7.2f us / %7.2f us\n",
               size_str, row.gdr.issue.median_us, row.gdr.issue.p99_us);
    }
}

static void print_bw_table(const char* title, const std::vector<DirectionRow>& rows) {
    printf("\n--- %s ---\n", title);
    printf("%-12s | %-18s\n", "Size", "    GDR (BW)     ");
    printf("%-12s-+-%-18s\n", "------------", "------------------");

    for (const auto& row : rows) {
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));
        printf("%-12s | %8.2f GB/s      \n", size_str, row.gdr.transfer.bw_gbs);
    }
}

static unsigned long long total_measured_ops(const std::vector<size_t>& sizes) {
    return static_cast<unsigned long long>(sizes.size()) *
           static_cast<unsigned long long>(ITERS) * 2ULL;
}

static double total_measured_gib(const std::vector<size_t>& sizes) {
    unsigned long long total_bytes = 0;
    for (size_t bytes : sizes)
        total_bytes += static_cast<unsigned long long>(bytes) * static_cast<unsigned long long>(ITERS) * 2ULL;
    return static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30);
}

static size_t allocate_large_gpu_region_or_die(int gpu_id, void** gpu_ptr, size_t min_bytes) {
    *gpu_ptr = nullptr;
    die_cuda(cudaSetDevice(gpu_id), "cudaSetDevice", gpu_id, 0);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    die_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", gpu_id, 0);

    size_t target_bytes = free_bytes;
    if (target_bytes > GPU_MR_RESERVE_BYTES)
        target_bytes -= GPU_MR_RESERVE_BYTES;
    else
        target_bytes = (target_bytes * 9) / 10;

    if (target_bytes < min_bytes)
        target_bytes = min_bytes;

    for (size_t attempt = target_bytes; attempt >= min_bytes; ) {
        cudaError_t ce = cudaMalloc(gpu_ptr, attempt);
        if (ce == cudaSuccess)
            return attempt;

        (void)cudaGetLastError();
        *gpu_ptr = nullptr;

        if (attempt == min_bytes)
            break;
        if (attempt <= min_bytes + GPU_MR_STEP_BYTES)
            attempt = min_bytes;
        else
            attempt -= GPU_MR_STEP_BYTES;
    }

    fprintf(stderr,
            "[full-gpu-mr] unable to allocate a large GPU region: gpu=%d min_bytes=%zu\n",
            gpu_id, min_bytes);
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <gpu_id> <nic_name>\n", argv[0]);
        fprintf(stderr, "Example: %s 4 mlx5_4\n", argv[0]);
        return 1;
    }

    const int gpu_id = std::atoi(argv[1]);
    const std::string nic_name = argv[2];

    printf("=================================================================\n");
    printf("  GDR Full-GPU-MR Benchmark  -  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    die_cuda(cudaSetDevice(gpu_id), "cudaSetDevice", gpu_id, 0);
    cudaDeviceProp prop{};
    die_cuda(cudaGetDeviceProperties(&prop, gpu_id), "cudaGetDeviceProperties", gpu_id, 0);
    printf("GPU: %s\n", prop.name);

    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= MAX_SWEEP_BYTES; s *= 4)
        sizes.push_back(s);

    void* gpu_big = nullptr;
    const size_t gpu_mr_bytes = allocate_large_gpu_region_or_die(gpu_id, &gpu_big, MAX_SWEEP_BYTES);
    void* h_src = nullptr;
    void* h_dst = nullptr;
    die_cuda(cudaHostAlloc(&h_src, MAX_SWEEP_BYTES, cudaHostAllocPortable), "cudaHostAlloc(h_src)", gpu_id, MAX_SWEEP_BYTES);
    die_cuda(cudaHostAlloc(&h_dst, MAX_SWEEP_BYTES, cudaHostAllocPortable), "cudaHostAlloc(h_dst)", gpu_id, MAX_SWEEP_BYTES);
    std::memset(h_src, 0xA5, MAX_SWEEP_BYTES);
    std::memset(h_dst, 0, MAX_SWEEP_BYTES);
    die_cuda(cudaMemset(gpu_big, 0x5A, MAX_SWEEP_BYTES), "cudaMemset(gpu_big)", gpu_id, MAX_SWEEP_BYTES);

    std::shared_ptr<GDRCopyChannel> ch;
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to open GDR channel: %s\n", e.what());
        return 1;
    }

    pin_windows_or_die(ch, h_src, MAX_SWEEP_BYTES, gpu_big, gpu_mr_bytes, "h2d_full_gpu_mr");
    ch->reset_stats();

    printf("Pinned GPU MR bytes: %.2f GiB\n", static_cast<double>(gpu_mr_bytes) / static_cast<double>(1ULL << 30));
    printf("Warmup=%d  Iters=%d\n", WARMUP, ITERS);
    printf("Benchmark mode: pre-register one large GPU MR, then measure GDR issue latency and sync-drain bandwidth\n");

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());
    for (size_t bytes : sizes)
        h2d_rows.push_back(DirectionRow{bytes, run_gdr_timings(ch, gpu_big, h_src, bytes, GDR_H2D, WARMUP, ITERS)});

    print_issue_table("Host->Device Issue Latency", h2d_rows);
    print_bw_table("Host->Device Bandwidth", h2d_rows);

    ch.reset();
    GDRCopyLib::shutdown();
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to reopen GDR channel: %s\n", e.what());
        return 1;
    }

    pin_windows_or_die(ch, h_dst, MAX_SWEEP_BYTES, gpu_big, gpu_mr_bytes, "d2h_full_gpu_mr");
    ch->reset_stats();

    std::vector<DirectionRow> d2h_rows;
    d2h_rows.reserve(sizes.size());
    for (size_t bytes : sizes)
        d2h_rows.push_back(DirectionRow{bytes, run_gdr_timings(ch, h_dst, gpu_big, bytes, GDR_D2H, WARMUP, ITERS)});

    print_issue_table("Device->Host Issue Latency", d2h_rows);
    print_bw_table("Device->Host Bandwidth", d2h_rows);

    printf("\n=================================================================\n");
    printf("Total measured ops: %llu\n", total_measured_ops(sizes));
    printf("Total measured bytes: %.2f GiB\n", total_measured_gib(sizes));
    printf("=================================================================\n");

    ch.reset();
    GDRCopyLib::shutdown();
    cudaFreeHost(h_src);
    cudaFreeHost(h_dst);
    cudaFree(gpu_big);
    return 0;
}
