/// If this global is present and has this value, then the
/// module is considered to be linked with the RT library.
__device__ __constant__ unsigned int __kerma_rt_linked__ = 0xFEEDC0DE;

/// Ultimately we want to minimize the complexity of each stub as
/// much as possible. For instance to remove branches and let the
/// insrumentation pass decide which one to invoke. This has the
/// advantage of improving runtime performance by peforming the
/// check at compile time, but increases code size. This is an
//// acceptable tradeoff for now.

#define __kerma_mem_access_type__ unsigned char

__device__ volatile bool __kerma_trace_status__ = true;

// We insert this call at the start of each kernel and then
// pass the result to every trace hook
extern "C" __device__ bool __kerma_trace_status() {
  bool status = __kerma_trace_status__;
  return status;
}

// We insert this call at the end of the kernel so that
// when the kernel is invoked again, the trace is not repeated
extern "C" __device__ void __kerma_stop_tracing() {
  __syncthreads();
  if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    __kerma_trace_status__ = false;
}

extern "C" __device__ void __kerma_rec_kernel(bool should_trace,
                                              unsigned char id,
                                              const char *name) {
  if ( !should_trace) return;

  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf("%%K.%d.%s\n", id, name);
  }
}

extern "C" __device__ void __kerma_rec_base(bool should_trace,
                                            unsigned char kernelid,
                                            const char *symbol,
                                            unsigned char addrspace,
                                            unsigned long baseaddr) {

  if ( !should_trace) return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  if (linearBlockIdx == 0 && threadIdx.x == 0 && threadIdx.y == 0 &&
      threadIdx.z == 0) {
    printf("%%B.%u.<%d>%s@%lu\n", kernelid, addrspace, symbol, baseaddr);
  }
}

extern "C" __device__ void __kerma_rec_access_b(bool should_trace,
                                                __kerma_mem_access_type__ ty,
                                                unsigned int bid,
                                                unsigned int line, unsigned int col,
                                                const char *name,
                                                unsigned long offset, unsigned int size) {
  if ( !should_trace) return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty,
                                            bid,
                                            linearLocalThreadIdx,
                                            line,
                                            col,
                                            name,
                                            offset,
                                            size);
  }
}

/// Record all threads in a specific warp in a specific block
/// The warp id must exist otherwise threads will perform
/// unecessary work.
extern "C" __device__ void __kerma_rec_access_w(bool should_trace,
                                                __kerma_mem_access_type__ ty,
                                                unsigned int bid,  unsigned int wid,
                                                unsigned int line, unsigned int col,
                                                const char *name, unsigned long offset,
                                                unsigned int size) {

  if ( !should_trace) return;

  unsigned int linearBlockIdx =
    blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int warpid = linearLocalThreadIdx / 32;

    if ( warpid == wid) {
      printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty,
                                              bid,
                                              linearLocalThreadIdx,
                                              line,
                                              col,
                                              name,
                                              offset,
                                              size);
    }
  }
}

/// Record a specific thread in a block
extern "C" __device__ void __kerma_rec_access_t(bool should_trace,
                                                __kerma_mem_access_type__ ty,
                                                unsigned int bid, unsigned int tid,
                                                unsigned int line, unsigned int col,
                                                const char *name, unsigned long offset,
                                                unsigned int size) {

  if ( !should_trace) return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if (linearLocalThreadIdx == tid)
      printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty,
                                              bid,
                                              tid,
                                              line,
                                              col,
                                              name,
                                              offset,
                                              size);
  }
}

// Record a memcopy. If a call to these functions is inserted, it means that
// at least one of the operands (source or dest) are in an address space
// other than local.
// If fromName or toName are NULL, this means that they are in local memory
// If one of the operands is in local memory, the offset is ignored and the
// record reports "loc"
// Note that the record does not include address space info. The address space
// for a (non-local) array is included in the %B records


/// Record a copy for all threads in a block
extern "C" __device__ void __kerma_rec_copy_b(bool should_trace,
                                              unsigned int bid,
                                              unsigned int line, unsigned int col,
                                              const char *sname, unsigned long soffset,
                                              const char *dname, unsigned long doffset,
                                              unsigned int size) {
  if ( !should_trace) return;

  unsigned int linearBlockIdx =
    blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if ( sname && dname) {
      printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                     line, col,
                                                     sname, soffset,
                                                     dname, doffset,
                                                     size);
    }
    else if ( sname) {
      printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
                                                 line, col,
                                                 sname, soffset,
                                                 size);
    }
    else {
      printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                 line, col,
                                                 dname, doffset,
                                                 size);
    }
  }
}

/// Record a copy for all threads in a specific warp in a specific block
extern "C" __device__ void __kerma_rec_copy_w(bool should_trace,
                                              unsigned int bid, unsigned int wid,
                                              unsigned int line, unsigned int col,
                                              const char *sname, unsigned long soffset,
                                              const char *dname, unsigned long doffset,
                                              unsigned int size) {
  if ( !should_trace) return;

  unsigned int linearBlockIdx =
    blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int warpid = linearLocalThreadIdx / 32;

    if ( warpid == wid) {
      if ( sname && dname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                       line, col,
                                                       sname, soffset,
                                                       dname, doffset,
                                                       size);
      }
      else if ( sname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
                                                   line, col,
                                                   sname, soffset,
                                                   size);
      }
      else {
        printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                   line, col,
                                                   dname, doffset,
                                                   size);
      }
    }
  }
}

/// Record a copy for all threads in a specific warp in a specific block
extern "C" __device__ void __kerma_rec_copy_t(bool should_trace,
                                              unsigned int bid, unsigned int tid,
                                              unsigned int line, unsigned int col,
                                              const char *sname, unsigned long soffset,
                                              const char *dname, unsigned long doffset,
                                              unsigned int size) {
  if ( !should_trace) return;

  unsigned int linearBlockIdx =
    blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if ( linearLocalThreadIdx == tid) {
      if ( sname && dname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                       line, col,
                                                       sname, soffset,
                                                       dname, doffset,
                                                       size);
      }
      else if ( sname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
                                                   line, col,
                                                   sname, soffset,
                                                   size);
      }
      else {
        printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
                                                   line, col,
                                                   dname, doffset,
                                                   size);
      }
    }
  }
}
