//
// Created by marco on 21/06/22.
//

#include "AutoCallableOptionCPU.cuh"
#include "../Shared/SharedAutoCallable.cuh"
#include "../../../StatisticUtils/StatisticsCPU.cuh"

#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>

AutoCallableOptionCPU::AutoCallableOptionCPU(Asset *asset, float rebase, const std::vector<float> &observationDates,
                                             const std::vector<float> &barriers, const std::vector<float> &payoffs,
                                             MonteCarloParams *monteCarloParams) : AutoCallableOption(asset, rebase,
                                                                                                      observationDates,
                                                                                                      barriers,
                                                                                                      payoffs),
                                                                                   monteCarloParams(monteCarloParams) {}

MonteCarloParams *AutoCallableOptionCPU::getMonteCarloParams() const {
    return monteCarloParams;
}

void AutoCallableOptionCPU::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    this->monteCarloParams = monteCarloParams;
}

SimulationResult AutoCallableOptionCPU::callPayoff() {
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(sizeof(float) * N_SIMULATIONS);

    size_t size = sizeof(float) * N_SIMULATIONS * observationDates.size();
    float *h_normals = (float *) malloc(size);
    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, monteCarloParams->getRngType());
    curandSetPseudoRandomGeneratorSeed(generator, monteCarloParams->getSeed());
    curandGenerateNormal(generator, d_normals, N_SIMULATIONS * observationDates.size(), 0.0f, 1.0f);

    cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    float *ptr_observationDates = observationDates.data();
    float *ptr_barriers = barriers.data();
    float *ptr_payoffs = payoffs.data();

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for(int i=0; i<N_SIMULATIONS; i++){
        autoCallablePayoff(getAsset()->getSpotPrice(),
                           getAsset()->getRiskFreeRate(),
                           getAsset()->getVolatility(),
                           rebase,
                           samples,
                           h_normals,
                           ptr_observationDates,
                           ptr_barriers,
                           ptr_payoffs,
                           observationDates.size(),
                           i, N_SIMULATIONS);
    }

    StatisticsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(), statistics.getConfidence(), statistics.getStdError(), elapsedTime);

    free(h_normals);
    curandDestroyGenerator(generator);
    cudaFree(d_normals);

    return result;
}

SimulationResult AutoCallableOptionCPU::putPayoff() {}