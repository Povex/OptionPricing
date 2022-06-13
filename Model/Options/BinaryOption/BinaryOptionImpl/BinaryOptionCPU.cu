//
// Created by marco on 31/05/22.
//

#include <chrono>
#include "BinaryOptionCPU.cuh"
#include "../../../StatisticUtils/StatisticUtilsCPU.cuh"
#include "../../Shared/SharedFunctions.cuh"
#include "../Shared/SharedBinaryOption.cuh"

BinaryOptionCPU::BinaryOptionCPU(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase,
                                 MonteCarloParams *monteCarloParams) : BinaryOption(asset, timeToMaturity, barrier,
                                                                                    payoff, rebase),
                                                                       monteCarloParams(monteCarloParams) {}

SimulationResult BinaryOptionCPU::callPayoff() {
    const unsigned int N_SIMULATIONS = monteCarloParams->getNSimulations();

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

    for (int i = 0; i<N_SIMULATIONS; i++) {
        binaryOptionSample(getAsset()->getSpotPrice(),
                                     getAsset()->getRiskFreeRate(),
                                     getAsset()->getVolatility(),
                                   timeToMaturity,
                                   barrier,
                                   h_normals,
                                   samples,
                                   i);
    }

    StatisticUtilsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(), statistics.getConfidence(),statistics.getStdError(), elapsedTime);

    free(h_normals); cudaFree(d_normals);
    free(samples);

    return result;
}

SimulationResult BinaryOptionCPU::putPayoff() {
    SimulationResult result = callPayoff();
    result.setValue(1 - result.getValue());

    return result;
}
