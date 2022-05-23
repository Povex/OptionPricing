//
// Created by marco on 23/05/22.
//


#include "EuropeanOptionSerialCPU.cuh"
#include "SharedFunctions.cuh"

#include "../../Utilities/StatisticUtils/StatisticUtilsCPU.cuh"

#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <numeric>

EuropeanOptionSerialCPU::EuropeanOptionSerialCPU(Asset *asset, float strikePrice, float timeToMaturity,
                                                 MonteCarloParams *monteCarloParams, GPUParams *gpuParams)
        : EuropeanOption(asset, strikePrice, timeToMaturity), monteCarloParams(monteCarloParams),
          gpuParams(gpuParams) {}

MonteCarloParams *EuropeanOptionSerialCPU::getMonteCarloParams() const {
    return monteCarloParams;
}

void EuropeanOptionSerialCPU::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    EuropeanOptionSerialCPU::monteCarloParams = monteCarloParams;
}

GPUParams *EuropeanOptionSerialCPU::getGpuParams() const {
    return gpuParams;
}

void EuropeanOptionSerialCPU::setGpuParams(GPUParams *gpuParams) {
    EuropeanOptionSerialCPU::gpuParams = gpuParams;
}

SimulationResult EuropeanOptionSerialCPU::callPayoff() {
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(N_SIMULATIONS * sizeof (float));

    size_t size = sizeof(float) * N_SIMULATIONS;
    float *h_normals = (float *) malloc(size);
    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42ULL);
    curandGenerateNormal(generator, d_normals, N_SIMULATIONS, 0.0f, 1.0f);
    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    cudaFree(d_normals);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    float S_T;
    for(int i=0; i<N_SIMULATIONS;i++){
        S_T = generateS_T(getAsset()->getSpotPrice(), getAsset()->getRiskFreeRate(),
                          getAsset()->getVolatility(), timeToMaturity, h_normals[i]);
        samples[i] = discountCall(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
    }

    StatisticUtilsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(), statistics.getConfidence(),statistics.getStdError(), elapsedTime);

    free(h_normals);

    return result;
}

SimulationResult EuropeanOptionSerialCPU::putPayoff() {
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(N_SIMULATIONS * sizeof (float));

    size_t size = sizeof(float) * N_SIMULATIONS;
    float *h_normals = (float *) malloc(size);
    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42ULL);
    curandGenerateNormal(generator, d_normals, N_SIMULATIONS, 0.0f, 1.0f);
    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    cudaFree(d_normals);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    float S_T;
    for(int i=0; i<N_SIMULATIONS;i++){
        S_T = generateS_T(getAsset()->getSpotPrice(), getAsset()->getRiskFreeRate(),
                          getAsset()->getVolatility(), timeToMaturity, h_normals[i]);
        samples[i] = discountPut(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
    }

    StatisticUtilsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(),
                            statistics.getConfidence(),
                            statistics.getStdError(),
                            elapsedTime);

    free(h_normals);

    return result;
}

