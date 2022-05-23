//
// Created by marco on 23/05/22.
//

#include "EuropeanOptionGPU.cuh"
#include "SharedFunctions.cuh"

#include "../../Utilities/errorHandler.cu"
#include "../../Utilities/StatisticUtils/StatisticUtilsGPU.cuh"

#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

// ------------------------------- BEGIN CUDA FUNCTIONS -------------------------------

__global__
void europeanCallPayoff(float spotPrice,
                        float riskFreeRate,
                        float volatility,
                        float timeToMaturity,
                        float strikePrice,
                        float *d_samples,
                        float *d_normals,
                        const unsigned int n_paths){

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S_T;
    while(i < n_paths) {
        S_T = generateS_T(spotPrice, riskFreeRate, volatility, timeToMaturity, d_normals[i]);
        d_samples[i] = discountCall(riskFreeRate, timeToMaturity, S_T, strikePrice);

        i += blockDim.x * gridDim.x;
    }
}

__global__
void europeanPutPayoff(float spotPrice,
                        float riskFreeRate,
                        float volatility,
                        float timeToMaturity,
                        float strikePrice,
                        float *d_samples,
                        float *d_normals,
                        const unsigned int n_paths){

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S_T;
    while(i < n_paths) {
        S_T = generateS_T(spotPrice, riskFreeRate, volatility, timeToMaturity, d_normals[i]);
        d_samples[i] = discountPut(riskFreeRate, timeToMaturity, S_T, strikePrice);

        i += blockDim.x * gridDim.x;
    }
}

// ----------------------------------- END CUDA FUNCTIONS --------------------------------


MonteCarloParams *EuropeanOptionGPU::getMonteCarloParams() const {
    return monteCarloParams;
}

void EuropeanOptionGPU::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    this->monteCarloParams = monteCarloParams;
}

GPUParams *EuropeanOptionGPU::getGpuParams() const {
    return gpuParams;
}

void EuropeanOptionGPU::setGpuParams(GPUParams *gpuParams) {
    this->gpuParams = gpuParams;
}

SimulationResult EuropeanOptionGPU::callPayoff() {
    // Initialize GPU params
    const int N_SIMULATION = getMonteCarloParams()->getNSimulations();
    dim3 blockDim1D(getGpuParams()->getNThreads());
    dim3 gridDim1D(std::ceil(float(getMonteCarloParams()->getNSimulations())/float(blockDim1D.x)));

    // Compute capability 2.1 needs this
    const int GPU_MAX_BLOCKS = 65535;
    const int GPU_MAX_THREADS = GPU_MAX_BLOCKS * 1024;
    if(N_SIMULATION > GPU_MAX_THREADS) gridDim1D = dim3(GPU_MAX_BLOCKS);

    // Initialize host-device vectors
    thrust::host_vector<float> h_samples(N_SIMULATION);
    thrust::device_vector<float> d_samples = h_samples;
    size_t size = sizeof(float) * N_SIMULATION;
    thrust::device_vector<float> d_normals(N_SIMULATION);
    float *ptr_normals = thrust::raw_pointer_cast(d_normals.data());
    float *ptr_samples = thrust::raw_pointer_cast(d_samples.data());

    // Create PRNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42ULL);
    curandGenerateNormal(generator, ptr_normals, N_SIMULATION, 0.0f, 1.0f);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Calculate payoff in device
    europeanCallPayoff<<<gridDim1D,blockDim1D>>>(getAsset()->getSpotPrice(),
                                                 getAsset()->getRiskFreeRate(),
                                                 getAsset()->getVolatility(),
                                                 timeToMaturity,
                                                 strikePrice,
                                                 ptr_samples,
                                                 ptr_normals,
                                                 N_SIMULATION);
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
    SimulationResult result(statistics.getMean(), statistics.getConfidence(),statistics.getStdError() , elapsedTime);

    return result;
}


EuropeanOptionGPU::EuropeanOptionGPU(Asset *asset, float strikePrice, float timeToMaturity,
                                     MonteCarloParams *monteCarloParams, GPUParams *gpuParams) : EuropeanOption(asset,
                                                                                                                strikePrice,
                                                                                                                timeToMaturity),
                                                                                                 monteCarloParams(
                                                                                                         monteCarloParams),
                                                                                                 gpuParams(gpuParams) {}

SimulationResult EuropeanOptionGPU::putPayoff() {
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
    thrust::device_vector<float> d_normals(N_SIMULATION);
    float *ptr_normals = thrust::raw_pointer_cast(d_normals.data());
    float *ptr_samples = thrust::raw_pointer_cast(d_samples.data());

    // Create PRNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 42ULL);
    curandGenerateNormal(generator, ptr_normals, N_SIMULATION, 0.0f, 1.0f);

    // Start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Calculate payoff in device
    europeanPutPayoff<<<gridDim1D,blockDim1D>>>(getAsset()->getSpotPrice(),
                                                getAsset()->getRiskFreeRate(),
                                                getAsset()->getVolatility(),
                                                timeToMaturity,
                                                strikePrice,
                                                ptr_samples,
                                                ptr_normals,
                                                N_SIMULATION);
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

    return result;
}


