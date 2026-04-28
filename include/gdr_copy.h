/**
 * gdr_copy.h - GPUDirect RDMA copy library
 *
 * This library provides a small abstraction that can replace selected
 * cudaMemcpy H2D and D2H paths when a GPU and NIC share a suitable PCIe
 * topology. When the RDMA path is available, the NIC can access GPU memory
 * directly. Otherwise the implementation falls back to cudaMemcpy.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

// Copy direction. The values mirror cudaMemcpyKind for easier search/replace.
enum GDRCopyKind {
    GDR_H2D = 1,   // host -> GPU (cudaMemcpyHostToDevice)
    GDR_D2H = 2,   // GPU -> host (cudaMemcpyDeviceToHost)
    GDR_D2D = 3,   // GPU -> GPU  (falls back to cudaMemcpy)
};

// Per-operation statistics collected by a channel.
struct GDRStats {
    double last_latency_us   = 0.0;   // Wall time of the last completed copy, in us.
    double avg_latency_us    = 0.0;   // Running average latency, in us.
    uint64_t total_bytes     = 0;     // Total bytes transferred.
    uint64_t total_ops       = 0;     // Total copy calls completed.
    uint64_t rdma_ops        = 0;     // Operations served by the RDMA path.
    uint64_t fallback_ops    = 0;     // Operations that fell back to cudaMemcpy.
};

// Opaque channel bound to one GPU and one NIC.
class GDRCopyChannel {
public:
    virtual ~GDRCopyChannel() = default;

    /**
     * Submit a copy operation.
     *
     * The current implementation keeps memcpy() as a submit-only API so its
     * behavior matches memcpy_async().
     *
     * @param dst   Destination pointer.
     * @param src   Source pointer.
     * @param bytes Number of bytes to transfer.
     * @param kind  GDR_H2D, GDR_D2H, or GDR_D2D.
     * @return      0 on successful submission, or a negative errno-style code.
     */
    virtual int memcpy(void* dst, const void* src,
                       size_t bytes, GDRCopyKind kind) = 0;

    /**
     * Submit a copy operation and return immediately.
     *
     * Completion order is transport-driven and may be out of order.
     */
    virtual int memcpy_async(void* dst, const void* src,
                             size_t bytes, GDRCopyKind kind) = 0;

    /**
     * Submit one asynchronous request and return request metadata.
     *
     * req_id identifies the request in the completion path. expected_wcs is
     * the number of CQEs expected for this request and is always at least 1.
     */
    virtual int memcpy_async_tagged(void* dst, const void* src,
                                    size_t bytes, GDRCopyKind kind,
                                    uint64_t* req_id, int* expected_wcs) = 0;

    /**
     * Pin or reuse the host MR window for a benchmark region.
     *
     * This does not submit any transfer.
     */
    virtual int pin_host_window(void* ptr, size_t bytes) = 0;

    /**
     * Pin or reuse the GPU MR window for a benchmark region.
     *
     * This does not submit any transfer.
     */
    virtual int pin_gpu_window(void* ptr, size_t bytes) = 0;

    /**
     * Drop the current GPU MR window.
     *
     * The next GPU-side transfer will have to register GPU memory again. This
     * is intended for MR registration microbenchmarks.
     */
    virtual int clear_gpu_window() = 0;

    /**
     * Poll one completion token without blocking.
     *
     * Returns 0 and sets req_id when one request is fully completed.
     * Returns -EAGAIN when no request has completed yet.
     */
    virtual int poll_wc(uint64_t* req_id) = 0;

    /**
     * Make non-blocking progress on pending asynchronous work.
     *
     * Returns 0 when one request completion is observed.
     * Returns -EAGAIN when no request has completed yet.
     */
    virtual int sync() = 0;

    // Return accumulated statistics.
    virtual GDRStats stats() const = 0;

    // Reset accumulated statistics.
    virtual void reset_stats() = 0;

    // Return the GPU index used when this channel was opened.
    virtual int gpu_id() const = 0;

    // Return the NIC device name, for example "mlx5_0".
    virtual const std::string& nic_name() const = 0;
};

// Library entry point.
class GDRCopyLib {
public:
    /**
     * Create or retrieve a cached channel for a GPU and NIC pair.
     *
     * @param gpu_id   CUDA device ordinal.
     * @param nic_name RDMA device name reported by ibv_devinfo.
     * @param use_odp  Enable on-demand paging for GPU MR registration.
     * @throws std::runtime_error on initialization failure.
     */
    static std::shared_ptr<GDRCopyChannel>
    open(int gpu_id, const std::string& nic_name, bool use_odp = false);

    /**
     * Check whether the RDMA path appears to be available.
     */
    static bool probe(int gpu_id, const std::string& nic_name);

    /**
     * Close all cached channels.
     */
    static void shutdown();

private:
    GDRCopyLib() = delete;
};
