//
// Created by marco on 21/06/22.
//

#include "EuropeanOptionCPU.cuh"
#include "../../Shared/SharedFunctions.cuh"
#include "../../../StatisticUtils/StatisticsSerialCPU.cuh"
#include "../../../StatisticUtils/StatisticsCPU.cuh"

#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <omp.h>
#include <iostream>

EuropeanOptionCPU::EuropeanOptionCPU(Asset *asset, float strikePrice, float timeToMaturity,
                                     MonteCarloParams *monteCarloParams) : EuropeanOption(asset, strikePrice,
                                                                                          timeToMaturity),
                                                                           monteCarloParams(monteCarloParams) {}

MonteCarloParams *EuropeanOptionCPU::getMonteCarloParams() const {
    return monteCarloParams;
}

void EuropeanOptionCPU::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    EuropeanOptionCPU::monteCarloParams = monteCarloParams;
}

SimulationResult EuropeanOptionCPU::callPayoff() {
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(N_SIMULATIONS * sizeof (float));

    size_t size = sizeof(float) * N_SIMULATIONS;
    float *h_normals = (float *) malloc(size);
    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, monteCarloParams->getRngType());
    curandSetPseudoRandomGeneratorSeed(generator, monteCarloParams->getSeed());
    curandGenerateNormal(generator, d_normals, N_SIMULATIONS, 0.0f, 1.0f);
    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    #pragma omp parallel default(shared)
    {
        float S_T;
        #pragma omp for
        for(int i=0; i<N_SIMULATIONS; i++){
            S_T = generateS_T(getAsset()->getSpotPrice(), getAsset()->getRiskFreeRate(),
                              getAsset()->getVolatility(), timeToMaturity, h_normals[i]);
            samples[i] = discountCall(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
        }
    }

    StatisticsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(),
                            statistics.getConfidence(),
                            statistics.getStdError(),
                            elapsedTime);

    free(h_normals); cudaFree(d_normals);
    free(samples);
    curandDestroyGenerator(generator);

    return result;
}

SimulationResult EuropeanOptionCPU::putPayoff() {
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(N_SIMULATIONS * sizeof (float));

    size_t size = sizeof(float) * N_SIMULATIONS;
    float *h_normals = (float *) malloc(size);
    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, monteCarloParams->getRngType());
    curandSetPseudoRandomGeneratorSeed(generator, monteCarloParams->getSeed());
    curandGenerateNormal(generator, d_normals, N_SIMULATIONS, 0.0f, 1.0f);
    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    #pragma omp parallel default(shared)
    {
        float S_T;
        #pragma omp for
        for(int i=0; i<N_SIMULATIONS; i++){
            S_T = generateS_T(getAsset()->getSpotPrice(), getAsset()->getRiskFreeRate(),
                              getAsset()->getVolatility(), timeToMaturity, h_normals[i]);
            samples[i] = discountPut(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
        }
    }

    StatisticsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(),
                            statistics.getConfidence(),
                            statistics.getStdError(),
                            elapsedTime);

    free(h_normals); cudaFree(d_normals);
    free(samples);
    curandDestroyGenerator(generator);

    return result;
}

