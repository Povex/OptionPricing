//
// Created by marco on 22/05/22.
//

#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../../Utilities/errorHandler.cu"

#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <numeric>

#include "AutocallableOption2.cuh"
#include "../../Utilities/StatisticUtils/StatisticUtilsCPU.cuh"
#include "../../Utilities/StatisticUtils/StatisticUtilsGPU.cuh"


__host__ __device__
void autoCallablePayoff(float spotPrice,
                              float riskFreeRate,
                              float volatility,
                              float rebase,
                              float *d_samples,
                              const float *d_normals,
                              const float *d_observationDates,
                              const float *d_barriers,
                              const float *d_payoffs,
                              int dateBarrierSize,
                              unsigned int i){
    bool barrier_hit = false;
    float S = spotPrice;
    int date_index = 0;
    float dt = d_observationDates[date_index];

    while (date_index <= dateBarrierSize - 1) {
        S = S * exp((riskFreeRate - (pow(volatility, 2) / 2)) * dt
                    + volatility* sqrt(dt) * d_normals[i + date_index]);

        if (S >= d_barriers[date_index]) { barrier_hit = true; break; }

        date_index++;
        dt = d_observationDates[date_index] - d_observationDates[date_index - 1];
    }

    if(!barrier_hit)
        d_samples[i] = exp(-riskFreeRate * d_observationDates[dateBarrierSize-1]) * rebase;
    else
        d_samples[i] = exp(-riskFreeRate * d_observationDates[date_index]) * d_payoffs[date_index];
}


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
    if(i >= n_paths) return;

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
            i);
}

__global__
void variance_samples_2(float *samples, int n, float mean){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < n) {
        samples[i] = powf(samples[i] - mean, 2);
        i += blockDim.x * gridDim.x;
    }
}

SimulationResult AutocallableOption2::callPayoffMontecarloCpu(){
    const int N_SIMULATIONS = getMonteCarloParams()->getNSimulations();
    float *samples = (float *)malloc(sizeof(float) * N_SIMULATIONS);

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

    float *ptr_observationDates = observationDates.data();
    float *ptr_barriers = barriers.data();
    float *ptr_payoffs = payoffs.data();

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

   for(unsigned int i=0; i<N_SIMULATIONS; i++){
       autoCallablePayoff( getAsset()->getSpotPrice(),
                           getAsset()->getRiskFreeRate(),
                                getAsset()->getVolatility(),
                                rebase,
                                samples,
                                h_normals,
                                ptr_observationDates,
                                ptr_barriers,
                                ptr_payoffs,
                                observationDates.size(),
                                i);

   }

    StatisticUtilsCPU statistics(samples, N_SIMULATIONS);
    statistics.calcMean();
    statistics.calcCI();

    // Calculate elapsed time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0f;

    SimulationResult result(statistics.getMean(), statistics.getConfidence(), statistics.getStdError(), elapsedTime);

    free(h_normals);

    return result;
}

SimulationResult AutocallableOption2::callPayoff() {
    // Initialize GPU params
    const int N_SIMULATION = getMonteCarloParams()->getNSimulations();
    dim3 blockDim1D(getGpuParams()->getNThreads());
    dim3 gridDim1D(std::ceil(float(getMonteCarloParams()->getNSimulations())/float(blockDim1D.x)));
    const int GPU_MAX_BLOCKS = 65535;
    const int GPU_MAX_THREADS = GPU_MAX_BLOCKS * 1024;
    if(N_SIMULATION > GPU_MAX_THREADS) gridDim1D = dim3(GPU_MAX_BLOCKS);

    // Initialize host-device vectors
    thrust::host_vector<float> h_samples(N_SIMULATION);
    thrust::device_vector<float> d_samples = h_samples;
    size_t size = sizeof(float) * N_SIMULATION;
    thrust::device_vector<float> d_normals(size * observationDates.size());
    float *ptr_normals = thrust::raw_pointer_cast(d_normals.data());
    float *ptr_samples = thrust::raw_pointer_cast(d_samples.data());

    // Create PRNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42ULL);
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

    KernelAutoCallableCallPayoff<<< gridDim1D, blockDim1D>>>(
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

    StatisticUtilsGPU statistics(blockDim1D, gridDim1D, d_samples);
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

SimulationResult AutocallableOption2::putPayoff() {
    return SimulationResult();
}

float AutocallableOption2::getRebase() const {
    return rebase;
}

void AutocallableOption2::setRebase(float rebase) {
    AutocallableOption2::rebase = rebase;
}


AutocallableOption2::AutocallableOption2(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams,
                                         float rebase, const std::vector<float> observationDates,
                                         const std::vector<float> barriers, const std::vector<float> payoffs)
        : Option(asset, gpuParams, monteCarloParams), rebase(rebase), observationDates(observationDates),
          barriers(barriers), payoffs(payoffs) {}


