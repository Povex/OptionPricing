//
// Created by marco on 24/05/22.
//

#include <chrono>
#include "AutoCallableOptionGPU.cuh"
#include "../../../Utilities/StatisticUtils/StatisticUtilsGPU.cuh"
#include "../Shared/SharedFunctions.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../../../Utilities/errorHandler.cu"

// --------------------------------------- BEGIN CUDA FUNCTIONS ---------------------------------


__global__ void KernelAutoCallableCallPayoff(float spotPrice,
                                             float riskFreeRate,
                                             float volatility,
                                             float rebase,
                                             float *d_samples,
                                             const float *d_normals,
                                             int n_paths,
                                             const float *d_observationDates,
                                             const float *d_barriers,
                                             const float *d_payoffs,
                                             int dateBarrierSize){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < n_paths) {
        autoCallablePayoff(spotPrice,
                           riskFreeRate,
                           volatility,
                           rebase,
                           d_samples,
                           d_normals,
                           d_observationDates,
                           d_barriers,
                           d_payoffs,
                           dateBarrierSize,
                           i,
                           n_paths);

        i += blockDim.x * gridDim.x;
    }
}

// ------------------------------------------ END CUDA FUNCTIONS ----------------------------------

AutoCallableOptionGPU::AutoCallableOptionGPU(Asset *asset, float rebase, const std::vector<float> &observationDates,
                                             const std::vector<float> &barriers, const std::vector<float> &payoffs,
                                             MonteCarloParams *monteCarloParams, GPUParams *gpuParams)
        : AutoCallableOption(asset, rebase, observationDates, barriers, payoffs), monteCarloParams(monteCarloParams),
          gpuParams(gpuParams) {}

MonteCarloParams *AutoCallableOptionGPU::getMonteCarloParams() const {
    return monteCarloParams;
}

void AutoCallableOptionGPU::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    this->monteCarloParams = monteCarloParams;
}

GPUParams *AutoCallableOptionGPU::getGpuParams() const {
    return gpuParams;
}

void AutoCallableOptionGPU::setGpuParams(GPUParams *gpuParams) {
    this->gpuParams = gpuParams;
}

SimulationResult AutoCallableOptionGPU::callPayoff() {
    // Initialize GPU params
    const int N_SIMULATION = getMonteCarloParams()->getNSimulations();

    // Initialize host-device vectors
    thrust::host_vector<float> h_samples(N_SIMULATION);
    thrust::device_vector<float> d_samples = h_samples;

    size_t size = sizeof(float) * N_SIMULATION * observationDates.size();
    thrust::device_vector<float> d_normals(N_SIMULATION * observationDates.size());
    float *ptr_normals = thrust::raw_pointer_cast(d_normals.data());
    float *ptr_samples = thrust::raw_pointer_cast(d_samples.data());

    // Create PRNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, monteCarloParams->getRngType());
    curandSetPseudoRandomGeneratorSeed(generator, monteCarloParams->getSeed());
    curandGenerateNormal(generator, ptr_normals,  N_SIMULATION * observationDates.size(), 0.0f, 1.0f);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    size_t observationSize = sizeof(float) * observationDates.size();

    float *d_observationDates;
    cudaMalloc((void**)&d_observationDates, observationSize);
    cudaMemcpy(d_observationDates, observationDates.data(), observationSize, cudaMemcpyHostToDevice);

    float *d_barriers;
    cudaMalloc((void**)&d_barriers, observationSize);
    cudaMemcpy(d_barriers, barriers.data(), observationSize, cudaMemcpyHostToDevice);

    float *d_payoffs;
    cudaMalloc((void**)&d_payoffs, observationSize);
    cudaMemcpy(d_payoffs, payoffs.data(), observationSize, cudaMemcpyHostToDevice);

    KernelAutoCallableCallPayoff<<< gpuParams->getBlocksPerGrid(), gpuParams->getThreadsPerBlock()>>>(
            getAsset()->getSpotPrice(),
            getAsset()->getRiskFreeRate(),
            getAsset()->getVolatility(),
            rebase,
            ptr_samples,
            ptr_normals,
            N_SIMULATION,
            d_observationDates,
            d_barriers,
            d_payoffs,
            observationDates.size()
    );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Clean memory from PRNG
    curandDestroyGenerator(generator);

    StatisticUtilsGPU statistics(gpuParams->getThreadsPerBlock(), gpuParams->getBlocksPerGrid(), d_samples);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    // Construct the simulation result
    SimulationResult result(statistics.getMean(), statistics.getConfidence(), statistics.getStdError(), elapsedTime);

    cudaFree(d_observationDates);
    cudaFree(d_barriers);
    cudaFree(d_payoffs);

    return result;
}

SimulationResult AutoCallableOptionGPU::putPayoff() {
    return SimulationResult();
}
