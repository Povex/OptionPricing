//
// Created by marco on 30/05/22.
//

#include "BinaryOptionGPU.cuh"
#include "../../EuropeanOption/Shared/SharedFunctions.cuh"
#include "../../../Utilities/errorHandler.cu"
#include "../../../Utilities/StatisticUtils/StatisticUtilsGPU.cuh"

#include <ctime>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>



__global__ void isOverBarrier(float spotPrice,
                                float riskFreeRate,
                                float volatility,
                                float timeToMaturity,
                                float barrier,
                                float *d_samples,
                                float *d_normals,
                                const unsigned int n_paths){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float stockT;

    while(i < n_paths){
        stockT = generateS_T(spotPrice,
                riskFreeRate,
                volatility,
                timeToMaturity,
                d_normals[i]);

        d_samples[i] = stockT >= barrier ? 1 : 0;

        i += gridDim.x * blockDim.x;
    }
}

BinaryOptionGPU::BinaryOptionGPU(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase,
                                 MonteCarloParams *monteCarloParams, GPUParams *gpuParams) : BinaryOption(asset,
                                                                                                          timeToMaturity,
                                                                                                          barrier,
                                                                                                          payoff,
                                                                                                          rebase),
                                                                                             monteCarloParams(
                                                                                                     monteCarloParams),
                                                                                             gpuParams(gpuParams) {}

SimulationResult BinaryOptionGPU::callPayoff() {
    int N_SIMULATION = monteCarloParams->getNSimulations();

    // Initialize host-device vectors
    thrust::host_vector<float> h_samples(N_SIMULATION);
    thrust::device_vector<float> d_samples = h_samples;
    size_t size = sizeof(float) * N_SIMULATION;
    thrust::device_vector<float> d_normals(N_SIMULATION);
    float *ptr_normals = thrust::raw_pointer_cast(d_normals.data());
    float *ptr_samples = thrust::raw_pointer_cast(d_samples.data());

    // Create PRNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, monteCarloParams->getRngType());
    curandSetPseudoRandomGeneratorSeed(generator, monteCarloParams->getSeed());
    curandGenerateNormal(generator, ptr_normals, N_SIMULATION, 0.0f, 1.0f);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Calculate payoff in device
    isOverBarrier<<<gpuParams->getBlocksPerGrid(),gpuParams->getThreadsPerBlock()>>>(
            getAsset()->getSpotPrice(),
            getAsset()->getRiskFreeRate(),
            getAsset()->getVolatility(),
            timeToMaturity,
            barrier,
            ptr_samples,
            ptr_normals,
            N_SIMULATION);
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
    SimulationResult result(statistics.getMean(), statistics.getConfidence(),statistics.getStdError() , elapsedTime);

    return result;

}

SimulationResult BinaryOptionGPU::putPayoff() {

}
