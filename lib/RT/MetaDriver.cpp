#include <stdio.h>
#include "Math.h"


void record(unsigned tidz, unsigned tidy, unsigned tidx, unsigned id, unsigned long offset) {
  printf("%u,%u,%u:%u:%lu\n", tidz, tidy, tidx, id, offset);
}

void block_loop(unsigned blockDimZ, unsigned blockDimY, unsigned blockDimX) {
  for ( unsigned z = 0; z < blockDimZ; ++z)
    for ( unsigned y = 0; y < blockDimY; ++y)
      for ( unsigned x = 0; x < blockDimX; ++x) {
        // meta_kernel(..., z, y, x);
      }
}

void warp_loop(unsigned blockDimZ, unsigned blockDimY, unsigned blockDimX, unsigned startingTidZ, unsigned startingTidY, unsigned startingTidX ) {
  unsigned int count = 0;
  for ( unsigned z = startingTidZ; z < blockDimZ; ++z)
    for ( unsigned y = startingTidY; y < blockDimY; ++y)
      for ( unsigned x = startingTidX; x < blockDimX; ++x) {
        // meta_kernel(..., z, y, x)
        if ( count == 31) return;
      }
}

void grid_loop(unsigned gridDimZ, unsigned gridDimY, unsigned gridDimX, unsigned blockDimZ, unsigned blockDimY, unsigned blockDimX) {
  for ( unsigned gz = 0; gz < blockDimZ; ++gz)
    for ( unsigned gy = 0; gy < blockDimY; ++gy)
      for ( unsigned gx = 0; gx < blockDimX; ++gx)
        for ( unsigned bz = 0; bz < blockDimZ; ++bz)
          for ( unsigned by = 0; by < blockDimY; ++by)
            for ( unsigned bx = 0; bx < blockDimX; ++bx) {
              
            }
}

int main(int argc, const char** argv) {
}