#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_of_warps()
{
    int gid = blockDim.x * gridDim.x * blockIdx.y +
            blockDim.x * blockIdx.x + threadIdx.x;

    int warp_id = threadIdx.x / 32;

    int gbid = gridDim.x * blockIdx.y + blockIdx.x;

    printf("tid.x: %02d, bid.x : %d, bid.y: %d, gid : %03d, warp_id: %d, gbid : %d\n",
           threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main()
{
    dim3 block(42);
    dim3 grid(2, 2);
    print_details_of_warps <<< grid, block >>> ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}

/*
tid.x: 00, bid.x : 0, bid.y: 0, gid : 000, warp_id: 0, gbid : 0
tid.x: 01, bid.x : 0, bid.y: 0, gid : 001, warp_id: 0, gbid : 0
tid.x: 02, bid.x : 0, bid.y: 0, gid : 002, warp_id: 0, gbid : 0
tid.x: 03, bid.x : 0, bid.y: 0, gid : 003, warp_id: 0, gbid : 0
tid.x: 04, bid.x : 0, bid.y: 0, gid : 004, warp_id: 0, gbid : 0
tid.x: 05, bid.x : 0, bid.y: 0, gid : 005, warp_id: 0, gbid : 0
tid.x: 06, bid.x : 0, bid.y: 0, gid : 006, warp_id: 0, gbid : 0
tid.x: 07, bid.x : 0, bid.y: 0, gid : 007, warp_id: 0, gbid : 0
tid.x: 08, bid.x : 0, bid.y: 0, gid : 008, warp_id: 0, gbid : 0
tid.x: 09, bid.x : 0, bid.y: 0, gid : 009, warp_id: 0, gbid : 0
tid.x: 10, bid.x : 0, bid.y: 0, gid : 010, warp_id: 0, gbid : 0
tid.x: 11, bid.x : 0, bid.y: 0, gid : 011, warp_id: 0, gbid : 0
tid.x: 12, bid.x : 0, bid.y: 0, gid : 012, warp_id: 0, gbid : 0
tid.x: 13, bid.x : 0, bid.y: 0, gid : 013, warp_id: 0, gbid : 0
tid.x: 14, bid.x : 0, bid.y: 0, gid : 014, warp_id: 0, gbid : 0
tid.x: 15, bid.x : 0, bid.y: 0, gid : 015, warp_id: 0, gbid : 0
tid.x: 16, bid.x : 0, bid.y: 0, gid : 016, warp_id: 0, gbid : 0
tid.x: 17, bid.x : 0, bid.y: 0, gid : 017, warp_id: 0, gbid : 0
tid.x: 18, bid.x : 0, bid.y: 0, gid : 018, warp_id: 0, gbid : 0
tid.x: 19, bid.x : 0, bid.y: 0, gid : 019, warp_id: 0, gbid : 0
tid.x: 20, bid.x : 0, bid.y: 0, gid : 020, warp_id: 0, gbid : 0
tid.x: 21, bid.x : 0, bid.y: 0, gid : 021, warp_id: 0, gbid : 0
tid.x: 22, bid.x : 0, bid.y: 0, gid : 022, warp_id: 0, gbid : 0
tid.x: 23, bid.x : 0, bid.y: 0, gid : 023, warp_id: 0, gbid : 0
tid.x: 24, bid.x : 0, bid.y: 0, gid : 024, warp_id: 0, gbid : 0
tid.x: 25, bid.x : 0, bid.y: 0, gid : 025, warp_id: 0, gbid : 0
tid.x: 26, bid.x : 0, bid.y: 0, gid : 026, warp_id: 0, gbid : 0
tid.x: 27, bid.x : 0, bid.y: 0, gid : 027, warp_id: 0, gbid : 0
tid.x: 28, bid.x : 0, bid.y: 0, gid : 028, warp_id: 0, gbid : 0
tid.x: 29, bid.x : 0, bid.y: 0, gid : 029, warp_id: 0, gbid : 0
tid.x: 30, bid.x : 0, bid.y: 0, gid : 030, warp_id: 0, gbid : 0
tid.x: 31, bid.x : 0, bid.y: 0, gid : 031, warp_id: 0, gbid : 0

tid.x: 32, bid.x : 0, bid.y: 0, gid : 032, warp_id: 1, gbid : 0
tid.x: 33, bid.x : 0, bid.y: 0, gid : 033, warp_id: 1, gbid : 0
tid.x: 34, bid.x : 0, bid.y: 0, gid : 034, warp_id: 1, gbid : 0
tid.x: 35, bid.x : 0, bid.y: 0, gid : 035, warp_id: 1, gbid : 0
tid.x: 36, bid.x : 0, bid.y: 0, gid : 036, warp_id: 1, gbid : 0
tid.x: 37, bid.x : 0, bid.y: 0, gid : 037, warp_id: 1, gbid : 0
tid.x: 38, bid.x : 0, bid.y: 0, gid : 038, warp_id: 1, gbid : 0
tid.x: 39, bid.x : 0, bid.y: 0, gid : 039, warp_id: 1, gbid : 0
tid.x: 40, bid.x : 0, bid.y: 0, gid : 040, warp_id: 1, gbid : 0
tid.x: 41, bid.x : 0, bid.y: 0, gid : 041, warp_id: 1, gbid : 0

tid.x: 00, bid.x : 1, bid.y: 0, gid : 042, warp_id: 0, gbid : 1
tid.x: 01, bid.x : 1, bid.y: 0, gid : 043, warp_id: 0, gbid : 1
tid.x: 02, bid.x : 1, bid.y: 0, gid : 044, warp_id: 0, gbid : 1
tid.x: 03, bid.x : 1, bid.y: 0, gid : 045, warp_id: 0, gbid : 1
tid.x: 04, bid.x : 1, bid.y: 0, gid : 046, warp_id: 0, gbid : 1
tid.x: 05, bid.x : 1, bid.y: 0, gid : 047, warp_id: 0, gbid : 1
tid.x: 06, bid.x : 1, bid.y: 0, gid : 048, warp_id: 0, gbid : 1
tid.x: 07, bid.x : 1, bid.y: 0, gid : 049, warp_id: 0, gbid : 1
tid.x: 08, bid.x : 1, bid.y: 0, gid : 050, warp_id: 0, gbid : 1
tid.x: 09, bid.x : 1, bid.y: 0, gid : 051, warp_id: 0, gbid : 1
tid.x: 10, bid.x : 1, bid.y: 0, gid : 052, warp_id: 0, gbid : 1
tid.x: 11, bid.x : 1, bid.y: 0, gid : 053, warp_id: 0, gbid : 1
tid.x: 12, bid.x : 1, bid.y: 0, gid : 054, warp_id: 0, gbid : 1
tid.x: 13, bid.x : 1, bid.y: 0, gid : 055, warp_id: 0, gbid : 1
tid.x: 14, bid.x : 1, bid.y: 0, gid : 056, warp_id: 0, gbid : 1
tid.x: 15, bid.x : 1, bid.y: 0, gid : 057, warp_id: 0, gbid : 1
tid.x: 16, bid.x : 1, bid.y: 0, gid : 058, warp_id: 0, gbid : 1
tid.x: 17, bid.x : 1, bid.y: 0, gid : 059, warp_id: 0, gbid : 1
tid.x: 18, bid.x : 1, bid.y: 0, gid : 060, warp_id: 0, gbid : 1
tid.x: 19, bid.x : 1, bid.y: 0, gid : 061, warp_id: 0, gbid : 1
tid.x: 20, bid.x : 1, bid.y: 0, gid : 062, warp_id: 0, gbid : 1
tid.x: 21, bid.x : 1, bid.y: 0, gid : 063, warp_id: 0, gbid : 1
tid.x: 22, bid.x : 1, bid.y: 0, gid : 064, warp_id: 0, gbid : 1
tid.x: 23, bid.x : 1, bid.y: 0, gid : 065, warp_id: 0, gbid : 1
tid.x: 24, bid.x : 1, bid.y: 0, gid : 066, warp_id: 0, gbid : 1
tid.x: 25, bid.x : 1, bid.y: 0, gid : 067, warp_id: 0, gbid : 1
tid.x: 26, bid.x : 1, bid.y: 0, gid : 068, warp_id: 0, gbid : 1
tid.x: 27, bid.x : 1, bid.y: 0, gid : 069, warp_id: 0, gbid : 1
tid.x: 28, bid.x : 1, bid.y: 0, gid : 070, warp_id: 0, gbid : 1
tid.x: 29, bid.x : 1, bid.y: 0, gid : 071, warp_id: 0, gbid : 1
tid.x: 30, bid.x : 1, bid.y: 0, gid : 072, warp_id: 0, gbid : 1
tid.x: 31, bid.x : 1, bid.y: 0, gid : 073, warp_id: 0, gbid : 1

tid.x: 32, bid.x : 1, bid.y: 0, gid : 074, warp_id: 1, gbid : 1
tid.x: 33, bid.x : 1, bid.y: 0, gid : 075, warp_id: 1, gbid : 1
tid.x: 34, bid.x : 1, bid.y: 0, gid : 076, warp_id: 1, gbid : 1
tid.x: 35, bid.x : 1, bid.y: 0, gid : 077, warp_id: 1, gbid : 1
tid.x: 36, bid.x : 1, bid.y: 0, gid : 078, warp_id: 1, gbid : 1
tid.x: 37, bid.x : 1, bid.y: 0, gid : 079, warp_id: 1, gbid : 1
tid.x: 38, bid.x : 1, bid.y: 0, gid : 080, warp_id: 1, gbid : 1
tid.x: 39, bid.x : 1, bid.y: 0, gid : 081, warp_id: 1, gbid : 1
tid.x: 40, bid.x : 1, bid.y: 0, gid : 082, warp_id: 1, gbid : 1
tid.x: 41, bid.x : 1, bid.y: 0, gid : 083, warp_id: 1, gbid : 1

tid.x: 00, bid.x : 0, bid.y: 1, gid : 084, warp_id: 0, gbid : 2
tid.x: 01, bid.x : 0, bid.y: 1, gid : 085, warp_id: 0, gbid : 2
tid.x: 02, bid.x : 0, bid.y: 1, gid : 086, warp_id: 0, gbid : 2
tid.x: 03, bid.x : 0, bid.y: 1, gid : 087, warp_id: 0, gbid : 2
tid.x: 04, bid.x : 0, bid.y: 1, gid : 088, warp_id: 0, gbid : 2
tid.x: 05, bid.x : 0, bid.y: 1, gid : 089, warp_id: 0, gbid : 2
tid.x: 06, bid.x : 0, bid.y: 1, gid : 090, warp_id: 0, gbid : 2
tid.x: 07, bid.x : 0, bid.y: 1, gid : 091, warp_id: 0, gbid : 2
tid.x: 08, bid.x : 0, bid.y: 1, gid : 092, warp_id: 0, gbid : 2
tid.x: 09, bid.x : 0, bid.y: 1, gid : 093, warp_id: 0, gbid : 2
tid.x: 10, bid.x : 0, bid.y: 1, gid : 094, warp_id: 0, gbid : 2
tid.x: 11, bid.x : 0, bid.y: 1, gid : 095, warp_id: 0, gbid : 2
tid.x: 12, bid.x : 0, bid.y: 1, gid : 096, warp_id: 0, gbid : 2
tid.x: 13, bid.x : 0, bid.y: 1, gid : 097, warp_id: 0, gbid : 2
tid.x: 14, bid.x : 0, bid.y: 1, gid : 098, warp_id: 0, gbid : 2
tid.x: 15, bid.x : 0, bid.y: 1, gid : 099, warp_id: 0, gbid : 2
tid.x: 16, bid.x : 0, bid.y: 1, gid : 100, warp_id: 0, gbid : 2
tid.x: 17, bid.x : 0, bid.y: 1, gid : 101, warp_id: 0, gbid : 2
tid.x: 18, bid.x : 0, bid.y: 1, gid : 102, warp_id: 0, gbid : 2
tid.x: 19, bid.x : 0, bid.y: 1, gid : 103, warp_id: 0, gbid : 2
tid.x: 20, bid.x : 0, bid.y: 1, gid : 104, warp_id: 0, gbid : 2
tid.x: 21, bid.x : 0, bid.y: 1, gid : 105, warp_id: 0, gbid : 2
tid.x: 22, bid.x : 0, bid.y: 1, gid : 106, warp_id: 0, gbid : 2
tid.x: 23, bid.x : 0, bid.y: 1, gid : 107, warp_id: 0, gbid : 2
tid.x: 24, bid.x : 0, bid.y: 1, gid : 108, warp_id: 0, gbid : 2
tid.x: 25, bid.x : 0, bid.y: 1, gid : 109, warp_id: 0, gbid : 2
tid.x: 26, bid.x : 0, bid.y: 1, gid : 110, warp_id: 0, gbid : 2
tid.x: 27, bid.x : 0, bid.y: 1, gid : 111, warp_id: 0, gbid : 2
tid.x: 28, bid.x : 0, bid.y: 1, gid : 112, warp_id: 0, gbid : 2
tid.x: 29, bid.x : 0, bid.y: 1, gid : 113, warp_id: 0, gbid : 2
tid.x: 30, bid.x : 0, bid.y: 1, gid : 114, warp_id: 0, gbid : 2
tid.x: 31, bid.x : 0, bid.y: 1, gid : 115, warp_id: 0, gbid : 2

tid.x: 32, bid.x : 0, bid.y: 1, gid : 116, warp_id: 1, gbid : 2
tid.x: 33, bid.x : 0, bid.y: 1, gid : 117, warp_id: 1, gbid : 2
tid.x: 34, bid.x : 0, bid.y: 1, gid : 118, warp_id: 1, gbid : 2
tid.x: 35, bid.x : 0, bid.y: 1, gid : 119, warp_id: 1, gbid : 2
tid.x: 36, bid.x : 0, bid.y: 1, gid : 120, warp_id: 1, gbid : 2
tid.x: 37, bid.x : 0, bid.y: 1, gid : 121, warp_id: 1, gbid : 2
tid.x: 38, bid.x : 0, bid.y: 1, gid : 122, warp_id: 1, gbid : 2
tid.x: 39, bid.x : 0, bid.y: 1, gid : 123, warp_id: 1, gbid : 2
tid.x: 40, bid.x : 0, bid.y: 1, gid : 124, warp_id: 1, gbid : 2
tid.x: 41, bid.x : 0, bid.y: 1, gid : 125, warp_id: 1, gbid : 2

tid.x: 00, bid.x : 1, bid.y: 1, gid : 126, warp_id: 0, gbid : 3
tid.x: 01, bid.x : 1, bid.y: 1, gid : 127, warp_id: 0, gbid : 3
tid.x: 02, bid.x : 1, bid.y: 1, gid : 128, warp_id: 0, gbid : 3
tid.x: 03, bid.x : 1, bid.y: 1, gid : 129, warp_id: 0, gbid : 3
tid.x: 04, bid.x : 1, bid.y: 1, gid : 130, warp_id: 0, gbid : 3
tid.x: 05, bid.x : 1, bid.y: 1, gid : 131, warp_id: 0, gbid : 3
tid.x: 06, bid.x : 1, bid.y: 1, gid : 132, warp_id: 0, gbid : 3
tid.x: 07, bid.x : 1, bid.y: 1, gid : 133, warp_id: 0, gbid : 3
tid.x: 08, bid.x : 1, bid.y: 1, gid : 134, warp_id: 0, gbid : 3
tid.x: 09, bid.x : 1, bid.y: 1, gid : 135, warp_id: 0, gbid : 3
tid.x: 10, bid.x : 1, bid.y: 1, gid : 136, warp_id: 0, gbid : 3
tid.x: 11, bid.x : 1, bid.y: 1, gid : 137, warp_id: 0, gbid : 3
tid.x: 12, bid.x : 1, bid.y: 1, gid : 138, warp_id: 0, gbid : 3
tid.x: 13, bid.x : 1, bid.y: 1, gid : 139, warp_id: 0, gbid : 3
tid.x: 14, bid.x : 1, bid.y: 1, gid : 140, warp_id: 0, gbid : 3
tid.x: 15, bid.x : 1, bid.y: 1, gid : 141, warp_id: 0, gbid : 3
tid.x: 16, bid.x : 1, bid.y: 1, gid : 142, warp_id: 0, gbid : 3
tid.x: 17, bid.x : 1, bid.y: 1, gid : 143, warp_id: 0, gbid : 3
tid.x: 18, bid.x : 1, bid.y: 1, gid : 144, warp_id: 0, gbid : 3
tid.x: 19, bid.x : 1, bid.y: 1, gid : 145, warp_id: 0, gbid : 3
tid.x: 20, bid.x : 1, bid.y: 1, gid : 146, warp_id: 0, gbid : 3
tid.x: 21, bid.x : 1, bid.y: 1, gid : 147, warp_id: 0, gbid : 3
tid.x: 22, bid.x : 1, bid.y: 1, gid : 148, warp_id: 0, gbid : 3
tid.x: 23, bid.x : 1, bid.y: 1, gid : 149, warp_id: 0, gbid : 3
tid.x: 24, bid.x : 1, bid.y: 1, gid : 150, warp_id: 0, gbid : 3
tid.x: 25, bid.x : 1, bid.y: 1, gid : 151, warp_id: 0, gbid : 3
tid.x: 26, bid.x : 1, bid.y: 1, gid : 152, warp_id: 0, gbid : 3
tid.x: 27, bid.x : 1, bid.y: 1, gid : 153, warp_id: 0, gbid : 3
tid.x: 28, bid.x : 1, bid.y: 1, gid : 154, warp_id: 0, gbid : 3
tid.x: 29, bid.x : 1, bid.y: 1, gid : 155, warp_id: 0, gbid : 3
tid.x: 30, bid.x : 1, bid.y: 1, gid : 156, warp_id: 0, gbid : 3
tid.x: 31, bid.x : 1, bid.y: 1, gid : 157, warp_id: 0, gbid : 3

tid.x: 32, bid.x : 1, bid.y: 1, gid : 158, warp_id: 1, gbid : 3
tid.x: 33, bid.x : 1, bid.y: 1, gid : 159, warp_id: 1, gbid : 3
tid.x: 34, bid.x : 1, bid.y: 1, gid : 160, warp_id: 1, gbid : 3
tid.x: 35, bid.x : 1, bid.y: 1, gid : 161, warp_id: 1, gbid : 3
tid.x: 36, bid.x : 1, bid.y: 1, gid : 162, warp_id: 1, gbid : 3
tid.x: 37, bid.x : 1, bid.y: 1, gid : 163, warp_id: 1, gbid : 3
tid.x: 38, bid.x : 1, bid.y: 1, gid : 164, warp_id: 1, gbid : 3
tid.x: 39, bid.x : 1, bid.y: 1, gid : 165, warp_id: 1, gbid : 3
tid.x: 40, bid.x : 1, bid.y: 1, gid : 166, warp_id: 1, gbid : 3
tid.x: 41, bid.x : 1, bid.y: 1, gid : 167, warp_id: 1, gbid : 3


*/
