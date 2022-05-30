//
// Created by marco on 24/05/22.
//

#include <strstream>
#include "ContextGPU.cuh"

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
            {
                    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
                    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
                    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
                    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
                    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
                    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
                    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
                    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
                    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
                    { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
                    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
                    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
                    {   -1, -1 }
            };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

ContextGPU* ContextGPU::pInstance = nullptr;

ContextGPU* ContextGPU::instance() {
    if(pInstance == nullptr){
        pInstance = new ContextGPU;
    }

    return pInstance;
}

ContextGPU::ContextGPU() {
    // set the first CUDA device available
    currentDevice = 0;

    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(currentDevice);
    cudaGetDeviceProperties(&deviceProp, currentDevice);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
}

void ContextGPU::printProperties(){
    printf("Starting...\n\n");
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
           (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    }

    printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
           deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
           deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
           deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


    printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
    printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
    printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
}

const cudaDeviceProp &ContextGPU::getDeviceProp() const {
    return deviceProp;
}



/* What is the best parameters configuration ?
 * https://forums.developer.nvidia.com/t/block-sizes-effect-on-program-performance-why-does-my-program-run-faster-at-seemingly-random-sizes/47069/2
 */
dim3 ContextGPU::getOptimalBlocksPerGrid(dim3 threadsPerBlock, unsigned int nSimulations) {
    int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock.x);

    if(nBlocksPerGrid > deviceProp.maxGridSize[0]){
        nBlocksPerGrid = deviceProp.maxGridSize[0];
    }

    return dim3(nBlocksPerGrid);
}

