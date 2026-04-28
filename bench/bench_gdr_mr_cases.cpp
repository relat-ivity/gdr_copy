/**
 * bench_gdr_mr_cases.cpp - extra GDR bandwidth cases around GPU MR placement
 *
 * Usage:
 *   sudo ./build/bench_gdr_mr_cases <gpu_id> <nic_name>
 *   sudo ./build/bench_gdr_mr_cases 4 mlx5_4
 *
 * This benchmark focuses on three GPU-MR-related bandwidth cases:
 *   1. GPU address is outside the pinned GPU MR window:
 *      use a GPU staging MR buffer and one extra D2D copy.
 *   2. GPU address is outside the staging MR buffer:
 *      pin the actual GPU buffer once via ibv_reg_mr and then GDR directly.
 *   3. Pin a 20 GiB HBM region as the GPU MR window and measure transfers
 *      whose GPU address falls inside that 20 GiB region.
 *
 * To avoid ambiguity in "copy to GPU MR buffer then GDR", this executable
 * reports both H2D and D2H bandwidth:
 *   - H2D staged case: GDR to GPU MR staging buffer, then D2D to actual dst
 *   - D2H staged case: D2D from actual src to GPU MR staging buffer, then GDR
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

static constexpr int WARMUP = 100;
static constexpr int ITERS  = 1000;
static constexpr size_t HBM_MR_BYTES  = 20ULL << 30;  // 20 GiB
static constexpr size_t HBM_MR_OFFSET = 8ULL  << 30;  // use an interior address

struct CaseRow {
    size_t bytes = 0;
    double staged_bw = 0.0;
    double reg_once_bw = 0.0;
    double hbm20_bw = -1.0;  // <0 means N/A
};

static double now_us() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

static void die_cuda(cudaError_t rc, const char* what, int gpu_id, size_t bytes) {
    if (rc == cudaSuccess)
        return;
    fprintf(stderr, "[mr-cases] %s failed: gpu=%d bytes=%zu err=%s\n",
            what, gpu_id, bytes, cudaGetErrorString(rc));
    std::exit(2);
}

static void format_size(size_t bytes, char* out, size_t out_len) {
    if (bytes < (1 << 10))      snprintf(out, out_len, "%zuB", bytes);
    else if (bytes < (1 << 20)) snprintf(out, out_len, "%zuKiB", bytes >> 10);
    else                        snprintf(out, out_len, "%zuMiB", bytes >> 20);
}

static void print_usage(const char* argv0) {
    fprintf(stderr, "Usage: %s <gpu_id> <nic_name>\n", argv0);
    fprintf(stderr, "Example: %s 4 mlx5_4\n", argv0);
}

static std::shared_ptr<GDRCopyChannel> reopen_channel(int gpu_id, const std::string& nic_name) {
    GDRCopyLib::shutdown();
    return GDRCopyLib::open(gpu_id, nic_name);
}

static void pin_windows_or_die(const std::shared_ptr<GDRCopyChannel>& ch,
                               void* host_ptr, size_t host_bytes,
                               void* gpu_ptr, size_t gpu_bytes,
                               const char* label)
{
    if (ch->pin_host_window(host_ptr, host_bytes) != 0) {
        fprintf(stderr, "[mr-cases] pin_host_window failed: case=%s bytes=%zu\n",
                label, host_bytes);
        std::exit(2);
    }
    if (ch->pin_gpu_window(gpu_ptr, gpu_bytes) != 0) {
        fprintf(stderr, "[mr-cases] pin_gpu_window failed: case=%s bytes=%zu\n",
                label, gpu_bytes);
        std::exit(2);
    }
}

static void submit_one_gdr_and_wait(const std::shared_ptr<GDRCopyChannel>& ch,
                                    void* dst, const void* src,
                                    size_t bytes, GDRCopyKind kind,
                                    const char* label)
{
    while (true) {
        int rc = ch->memcpy_async(dst, src, bytes, kind);
        if (rc == 0)
            break;
        if (rc == -EBUSY) {
            while (true) {
                int sc = ch->sync();
                if (sc == 0)
                    break;
                if (sc == -EAGAIN)
                    continue;
                fprintf(stderr,
                        "[mr-cases] gdr sync failed: case=%s rc=%d kind=%d bytes=%zu\n",
                        label, sc, (int)kind, bytes);
                std::exit(2);
            }
            continue;
        }
        fprintf(stderr,
                "[mr-cases] gdr memcpy_async failed: case=%s rc=%d kind=%d bytes=%zu\n",
                label, rc, (int)kind, bytes);
        std::exit(2);
    }

    while (true) {
        int sc = ch->sync();
        if (sc == 0)
            return;
        if (sc == -EAGAIN)
            continue;
        fprintf(stderr,
                "[mr-cases] gdr sync failed: case=%s rc=%d kind=%d bytes=%zu\n",
                label, sc, (int)kind, bytes);
        std::exit(2);
    }
}

static double run_direct_gdr_total_us(const std::shared_ptr<GDRCopyChannel>& ch,
                                      void* dst, const void* src,
                                      size_t bytes, GDRCopyKind kind,
                                      int count, const char* label)
{
    if (count <= 0)
        return 0.0;

    double t0 = now_us();
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
                        "[mr-cases] gdr sync failed: case=%s rc=%d kind=%d bytes=%zu done=%d issued=%d\n",
                        label, sc, (int)kind, bytes, done, issued);
                std::exit(2);
            }
            continue;
        }
        fprintf(stderr,
                "[mr-cases] gdr memcpy_async failed: case=%s rc=%d kind=%d bytes=%zu issued=%d/%d\n",
                label, rc, (int)kind, bytes, issued, count);
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
                "[mr-cases] gdr sync failed: case=%s rc=%d kind=%d bytes=%zu done=%d/%d\n",
                label, sc, (int)kind, bytes, done, count);
        std::exit(2);
    }

    return now_us() - t0;
}

static double run_direct_gdr_bw(const std::shared_ptr<GDRCopyChannel>& ch,
                                void* dst, const void* src,
                                size_t bytes, GDRCopyKind kind,
                                int warmup, int iters, const char* label)
{
    run_direct_gdr_total_us(ch, dst, src, bytes, kind, warmup, label);
    double total_us = run_direct_gdr_total_us(ch, dst, src, bytes, kind, iters, label);
    return (iters > 0 && total_us > 0.0)
             ? ((double)bytes * (double)iters / 1e9) / (total_us / 1e6)
             : 0.0;
}

static double run_staged_h2d_bw(const std::shared_ptr<GDRCopyChannel>& ch,
                                void* actual_gpu, void* stage_gpu, const void* host_src,
                                size_t bytes, int warmup, int iters,
                                cudaStream_t stream)
{
    auto one_iter = [&]() {
        submit_one_gdr_and_wait(ch, stage_gpu, host_src, bytes, GDR_H2D, "staged_h2d");
        cudaError_t ce = cudaMemcpyAsync(actual_gpu, stage_gpu, bytes,
                                         cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[mr-cases] cudaMemcpyAsync D2D failed: case=staged_h2d bytes=%zu err=%s\n",
                    bytes, cudaGetErrorString(ce));
            std::exit(2);
        }
        ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[mr-cases] cudaStreamSynchronize failed: case=staged_h2d bytes=%zu err=%s\n",
                    bytes, cudaGetErrorString(ce));
            std::exit(2);
        }
    };

    for (int i = 0; i < warmup; ++i)
        one_iter();

    double t0 = now_us();
    for (int i = 0; i < iters; ++i)
        one_iter();
    double total_us = now_us() - t0;

    return (iters > 0 && total_us > 0.0)
             ? ((double)bytes * (double)iters / 1e9) / (total_us / 1e6)
             : 0.0;
}

static double run_staged_d2h_bw(const std::shared_ptr<GDRCopyChannel>& ch,
                                void* host_dst, void* stage_gpu, const void* actual_gpu,
                                size_t bytes, int warmup, int iters,
                                cudaStream_t stream)
{
    auto one_iter = [&]() {
        cudaError_t ce = cudaMemcpyAsync(stage_gpu, actual_gpu, bytes,
                                         cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[mr-cases] cudaMemcpyAsync D2D failed: case=staged_d2h bytes=%zu err=%s\n",
                    bytes, cudaGetErrorString(ce));
            std::exit(2);
        }
        ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[mr-cases] cudaStreamSynchronize failed: case=staged_d2h bytes=%zu err=%s\n",
                    bytes, cudaGetErrorString(ce));
            std::exit(2);
        }
        submit_one_gdr_and_wait(ch, host_dst, stage_gpu, bytes, GDR_D2H, "staged_d2h");
    };

    for (int i = 0; i < warmup; ++i)
        one_iter();

    double t0 = now_us();
    for (int i = 0; i < iters; ++i)
        one_iter();
    double total_us = now_us() - t0;

    return (iters > 0 && total_us > 0.0)
             ? ((double)bytes * (double)iters / 1e9) / (total_us / 1e6)
             : 0.0;
}

static void print_case_table(const char* title, const std::vector<CaseRow>& rows) {
    printf("\n--- %s ---\n", title);
    printf("%-12s | %-18s | %-18s | %-18s\n",
           "Size", "stage->MR->GDR", "reg-once+GDR", "20GB-MR+GDR");
    printf("%-12s-+-%-18s-+-%-18s-+-%-18s\n",
           "------------", "------------------", "------------------", "------------------");

    for (const auto& row : rows) {
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));
        if (row.hbm20_bw >= 0.0) {
            printf("%-12s | %8.2f GB/s      | %8.2f GB/s      | %8.2f GB/s      \n",
                   size_str, row.staged_bw, row.reg_once_bw, row.hbm20_bw);
        } else {
            printf("%-12s | %8.2f GB/s      | %8.2f GB/s      | %8s          \n",
                   size_str, row.staged_bw, row.reg_once_bw, "N/A");
        }
    }
}

static bool try_allocate_big_gpu_buffer(int gpu_id, void** buf, size_t bytes) {
    *buf = nullptr;
    cudaError_t ce = cudaSetDevice(gpu_id);
    if (ce != cudaSuccess)
        return false;
    ce = cudaMalloc(buf, bytes);
    if (ce != cudaSuccess) {
        (void)cudaGetLastError();
        *buf = nullptr;
        return false;
    }
    return true;
}

static std::vector<CaseRow> run_h2d_suite(int gpu_id, const std::string& nic_name,
                                          const std::vector<size_t>& sizes)
{
    const size_t max_bytes = sizes.back();
    std::vector<CaseRow> rows(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i)
        rows[i].bytes = sizes[i];

    void* host_src = nullptr;
    void* gpu_stage = nullptr;
    void* gpu_actual = nullptr;
    void* gpu_big = nullptr;
    cudaStream_t stream{};

    die_cuda(cudaSetDevice(gpu_id), "cudaSetDevice", gpu_id, 0);
    die_cuda(cudaHostAlloc(&host_src, max_bytes, cudaHostAllocPortable),
             "cudaHostAlloc", gpu_id, max_bytes);
    die_cuda(cudaMalloc(&gpu_stage, max_bytes), "cudaMalloc(stage)", gpu_id, max_bytes);
    die_cuda(cudaMalloc(&gpu_actual, max_bytes), "cudaMalloc(actual)", gpu_id, max_bytes);
    std::memset(host_src, 0xA5, max_bytes);
    die_cuda(cudaMemset(gpu_stage, 0, max_bytes), "cudaMemset(stage)", gpu_id, max_bytes);
    die_cuda(cudaMemset(gpu_actual, 0, max_bytes), "cudaMemset(actual)", gpu_id, max_bytes);
    die_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags", gpu_id, 0);

    {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_src, max_bytes, gpu_stage, max_bytes, "h2d_staged");
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].staged_bw = run_staged_h2d_bw(ch, gpu_actual, gpu_stage, host_src,
                                                  sizes[i], WARMUP, ITERS, stream);
        }
    }

    {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_src, max_bytes, gpu_actual, max_bytes, "h2d_reg_once");
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].reg_once_bw = run_direct_gdr_bw(ch, gpu_actual, host_src, sizes[i],
                                                    GDR_H2D, WARMUP, ITERS, "h2d_reg_once");
        }
    }

    if (try_allocate_big_gpu_buffer(gpu_id, &gpu_big, HBM_MR_BYTES)) {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_src, max_bytes, gpu_big, HBM_MR_BYTES, "h2d_hbm20");
        uint8_t* gpu_big_inner = static_cast<uint8_t*>(gpu_big) + HBM_MR_OFFSET;
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].hbm20_bw = run_direct_gdr_bw(ch, gpu_big_inner, host_src, sizes[i],
                                                 GDR_H2D, WARMUP, ITERS, "h2d_hbm20");
        }
        cudaFree(gpu_big);
        gpu_big = nullptr;
    }

    cudaStreamDestroy(stream);
    cudaFree(gpu_stage);
    cudaFree(gpu_actual);
    cudaFreeHost(host_src);
    GDRCopyLib::shutdown();
    return rows;
}

static std::vector<CaseRow> run_d2h_suite(int gpu_id, const std::string& nic_name,
                                          const std::vector<size_t>& sizes)
{
    const size_t max_bytes = sizes.back();
    std::vector<CaseRow> rows(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i)
        rows[i].bytes = sizes[i];

    void* host_dst = nullptr;
    void* gpu_stage = nullptr;
    void* gpu_actual = nullptr;
    void* gpu_big = nullptr;
    cudaStream_t stream{};

    die_cuda(cudaSetDevice(gpu_id), "cudaSetDevice", gpu_id, 0);
    die_cuda(cudaHostAlloc(&host_dst, max_bytes, cudaHostAllocPortable),
             "cudaHostAlloc", gpu_id, max_bytes);
    die_cuda(cudaMalloc(&gpu_stage, max_bytes), "cudaMalloc(stage)", gpu_id, max_bytes);
    die_cuda(cudaMalloc(&gpu_actual, max_bytes), "cudaMalloc(actual)", gpu_id, max_bytes);
    std::memset(host_dst, 0, max_bytes);
    die_cuda(cudaMemset(gpu_stage, 0x11, max_bytes), "cudaMemset(stage)", gpu_id, max_bytes);
    die_cuda(cudaMemset(gpu_actual, 0x5A, max_bytes), "cudaMemset(actual)", gpu_id, max_bytes);
    die_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags", gpu_id, 0);

    {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_dst, max_bytes, gpu_stage, max_bytes, "d2h_staged");
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].staged_bw = run_staged_d2h_bw(ch, host_dst, gpu_stage, gpu_actual,
                                                  sizes[i], WARMUP, ITERS, stream);
        }
    }

    {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_dst, max_bytes, gpu_actual, max_bytes, "d2h_reg_once");
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].reg_once_bw = run_direct_gdr_bw(ch, host_dst, gpu_actual, sizes[i],
                                                    GDR_D2H, WARMUP, ITERS, "d2h_reg_once");
        }
    }

    if (try_allocate_big_gpu_buffer(gpu_id, &gpu_big, HBM_MR_BYTES)) {
        auto ch = reopen_channel(gpu_id, nic_name);
        pin_windows_or_die(ch, host_dst, max_bytes, gpu_big, HBM_MR_BYTES, "d2h_hbm20");
        uint8_t* gpu_big_inner = static_cast<uint8_t*>(gpu_big) + HBM_MR_OFFSET;
        for (size_t i = 0; i < sizes.size(); ++i) {
            rows[i].hbm20_bw = run_direct_gdr_bw(ch, host_dst, gpu_big_inner, sizes[i],
                                                 GDR_D2H, WARMUP, ITERS, "d2h_hbm20");
        }
        cudaFree(gpu_big);
        gpu_big = nullptr;
    }

    cudaStreamDestroy(stream);
    cudaFree(gpu_stage);
    cudaFree(gpu_actual);
    cudaFreeHost(host_dst);
    GDRCopyLib::shutdown();
    return rows;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    int gpu_id = std::atoi(argv[1]);
    std::string nic_name = argv[2];

    printf("=================================================================\n");
    printf("  GDR MR Cases Benchmark  -  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    cudaDeviceProp prop{};
    die_cuda(cudaGetDeviceProperties(&prop, gpu_id), "cudaGetDeviceProperties", gpu_id, 0);
    printf("GPU: %s\n", prop.name);
    printf("Warmup=%d  Iters=%d  HBM-MR=20 GiB\n\n", WARMUP, ITERS);

    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    auto h2d_rows = run_h2d_suite(gpu_id, nic_name, sizes);
    auto d2h_rows = run_d2h_suite(gpu_id, nic_name, sizes);

    print_case_table("Host->Device Bandwidth (MR Cases)", h2d_rows);
    print_case_table("Device->Host Bandwidth (MR Cases)", d2h_rows);

    return 0;
}
