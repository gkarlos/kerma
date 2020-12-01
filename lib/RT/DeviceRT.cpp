/// Ultimately we want to minimize the complexity of each stub as
/// much as possible. For instance to remove branches and let the
/// insrumentation pass decide which one to invoke. This has the
/// advantage of improving runtime performance by peforming the
/// check at compile time, but increases code size. This is an
//// acceptable tradeoff for now.

#define __kerma_mem_access_type__ unsigned char

/// If this global is present and has this value, then the
/// module is considered to be linked with the RT library.
__device__ __constant__ unsigned int __kerma_rt_linked__ = 0xFEEDC0DE;

/// Read the trace status for a kernel
/// This call is inserted at the start of each kernel the result
/// is is passed to every trace hook.
///
/// The function just reads the value and brings to local memory.
/// We don't really need a function to do that per se. The alternative
/// is to just an IRBuilder to insert the relevant code.
/// By having this function however we let codegen do that for us,
/// so we do not have to insert extra code in the Instrumenter.
extern "C" __device__ bool __kerma_trace_status(bool *stop_tracing,
                                                unsigned int kernel_id) {
  bool status = stop_tracing[kernel_id];
  return status;
}

/// Signal that a kernel should not be traced anymore
/// This call is inserted before every exit point for the kernel function
///
/// Similarly to __kerma_trace_status we put this function here to
/// minimize the Instrumenter code.
extern "C" __device__ void __kerma_stop_tracing(bool *stop_tracing,
                                                unsigned int kernel_id) {
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    stop_tracing[kernel_id] = true;
}

extern "C" __device__ void __rec_access_mat_b(bool stop_tracing, unsigned bid, unsigned access_id, unsigned long offset) {
  if ( stop_tracing) return;
  unsigned int linearBlockIdx =
    blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  if ( linearBlockIdx == bid)
    printf("%u,%u,%u:%u:%lu\n", threadIdx.z, threadIdx.y, threadIdx.x, access_id, offset);
}

extern "C" __device__ void __rec_access_mat_w(bool stop_tracing, unsigned bid, unsigned wid, unsigned access_id, unsigned long offset) {
  if (stop_tracing) return;
  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int warpid = linearLocalThreadIdx / 32;
    if ( warpid == wid)
      printf("%u,%u,%u:%u:%lu\n", threadIdx.z, threadIdx.y, threadIdx.x, access_id, offset);
  }
}

extern "C" __device__ void __rec_access_mat_t(bool stop_tracing, unsigned bid, unsigned tid, unsigned access_id, unsigned long offset) {
  if (stop_tracing) return;
  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;
    if (linearLocalThreadIdx == tid)
      printf("%u,%u,%u:%u:%lu\n", threadIdx.z, threadIdx.y, threadIdx.x, access_id, offset);
  }
}

///
extern "C" __device__ void
__kerma_rec_kernel(bool stop_tracing, unsigned char id, const char *name) {
  if (stop_tracing)
    return;

  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf("%%K.%d.%s<%u,%u,%u><%u,%u,%u>\n", id, name, gridDim.z, gridDim.y,
           gridDim.x, blockDim.z, blockDim.y, blockDim.x);
  }
}

extern "C" __device__ void
__kerma_rec_base(bool stop_tracing, unsigned char kernelid, const char *symbol,
                 unsigned char addrspace, unsigned long baseaddr) {

  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  if (linearBlockIdx == 0 && threadIdx.x == 0 && threadIdx.y == 0 &&
      threadIdx.z == 0) {
    printf("%%B.%u.<%d>%s@%lu\n", kernelid, addrspace, symbol, baseaddr);
  }
}

extern "C" __device__ void
__kerma_rec_access_b(bool stop_tracing, __kerma_mem_access_type__ ty,
                     unsigned int bid, unsigned int line, unsigned int col,
                     const char *name, unsigned long offset,
                     unsigned int size) {
  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty, bid, linearLocalThreadIdx, line,
           col, name, offset, size);
  }
}

/// Record all threads in a specific warp in a specific block
/// The warp id must exist otherwise threads will perform
/// unecessary work.
extern "C" __device__ void
__kerma_rec_access_w(bool stop_tracing, __kerma_mem_access_type__ ty,
                     unsigned int bid, unsigned int wid, unsigned int line,
                     unsigned int col, const char *name, unsigned long offset,
                     unsigned int size) {

  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int warpid = linearLocalThreadIdx / 32;

    if (warpid == wid) {
      printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty, bid, linearLocalThreadIdx,
             line, col, name, offset, size);
    }
  }
}

/// Record a specific thread in a block
extern "C" __device__ void
__kerma_rec_access_t(bool stop_tracing, __kerma_mem_access_type__ ty,
                     unsigned int bid, unsigned int tid, unsigned int line,
                     unsigned int col, const char *name, unsigned long offset,
                     unsigned int size) {

  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if (linearLocalThreadIdx == tid)
      printf("!%c(%u,%u,%u:%u):%s[%lu]:%u\n", ty, bid, tid, line, col, name,
             offset, size);
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
extern "C" __device__ void
__kerma_rec_copy_b(bool stop_tracing, unsigned int bid, unsigned int line,
                   unsigned int col, const char *sname, unsigned long soffset,
                   const char *dname, unsigned long doffset,
                   unsigned int size) {
  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if (sname && dname) {
      printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
             line, col, sname, soffset, dname, doffset, size);
    } else if (sname) {
      printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
             line, col, sname, soffset, size);
    } else {
      printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
             line, col, dname, doffset, size);
    }
  }
}

/// Record a copy for all threads in a specific warp in a specific block
extern "C" __device__ void
__kerma_rec_copy_w(bool stop_tracing, unsigned int bid, unsigned int wid,
                   unsigned int line, unsigned int col, const char *sname,
                   unsigned long soffset, const char *dname,
                   unsigned long doffset, unsigned int size) {
  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int warpid = linearLocalThreadIdx / 32;

    if (warpid == wid) {
      if (sname && dname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid,
               linearLocalThreadIdx, line, col, sname, soffset, dname, doffset,
               size);
      } else if (sname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
               line, col, sname, soffset, size);
      } else {
        printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
               line, col, dname, doffset, size);
      }
    }
  }
}

/// Record a copy for all threads in a specific warp in a specific block
extern "C" __device__ void
__kerma_rec_copy_t(bool stop_tracing, unsigned int bid, unsigned int tid,
                   unsigned int line, unsigned int col, const char *sname,
                   unsigned long soffset, const char *dname,
                   unsigned long doffset, unsigned int size) {
  if (stop_tracing)
    return;

  unsigned int linearBlockIdx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  if (linearBlockIdx == bid) {
    unsigned int linearLocalThreadIdx = threadIdx.z * blockDim.x * blockDim.y +
                                        threadIdx.y * blockDim.x + threadIdx.x;

    if (linearLocalThreadIdx == tid) {
      if (sname && dname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>%s[%lu]:%u\n", bid,
               linearLocalThreadIdx, line, col, sname, soffset, dname, doffset,
               size);
      } else if (sname) {
        printf("!C(%u,%u,%u:%u):%s[%lu]>loc:%u\n", bid, linearLocalThreadIdx,
               line, col, sname, soffset, size);
      } else {
        printf("!C(%u,%u,%u:%u):loc>%s[%lu]:%u\n", bid, linearLocalThreadIdx,
               line, col, dname, doffset, size);
      }
    }
  }
}
