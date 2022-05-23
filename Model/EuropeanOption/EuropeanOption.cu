//
// Created by marco on 19/05/22.
//

#include "EuropeanOption.cuh"

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

#include "../../Utilities/errorHandler.cu"
#include "../../Utilities/StatisticUtils/StatisticUtilsCPU.cuh"
#include "../../Utilities/StatisticUtils/StatisticUtilsGPU.cuh"


__host__ __device__
float discoutCall(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice){
    return expf(-riskFreeRate * timeToMaturity) * fmaxf(0, S_T - strikePrice);
}

__host__ __device__
float discoutPut(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice){
    return expf(-riskFreeRate * timeToMaturity) * fmaxf(0, strikePrice - S_T);
}

__host__ __device__
float generateS_T(float spotPrice,
                float riskFreeRate,
                float volatility,
                float timeToMaturity,
                float z){

    return spotPrice * expf((riskFreeRate - 0.5f * powf(volatility, 2))
                     * timeToMaturity + volatility * sqrtf(timeToMaturity) * z);
}

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
        d_samples[i] = discoutCall(riskFreeRate, timeToMaturity, S_T, strikePrice);

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
        d_samples[i] = discoutPut(riskFreeRate, timeToMaturity, S_T, strikePrice);

        i += blockDim.x * gridDim.x;
    }
}

__global__
void variance_samples(float *samples, int n, float mean){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < n) {
        samples[i] = powf(samples[i] - mean, 2);
        i += blockDim.x * gridDim.x;
    }
}

float EuropeanOption::getStrikePrice() const {
    return strikePrice;
}

void EuropeanOption::setStrikePrice(float strikePrice) {
    EuropeanOption::strikePrice = strikePrice;
}

float EuropeanOption::getTimeToMaturity() const {
    return timeToMaturity;
}

void EuropeanOption::setTimeToMaturity(float timeToMaturity) {
    EuropeanOption::timeToMaturity = timeToMaturity;
}

EuropeanOption::EuropeanOption(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams) : Option(asset,
                                                                                                                gpuParams,
                                                                                                                monteCarloParams) {

}

EuropeanOption::EuropeanOption(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams,
                               float strikePrice, float timeToMaturity) : Option(asset, gpuParams, monteCarloParams),
                                                                            strikePrice(strikePrice),
                                                                            timeToMaturity(timeToMaturity) {}

EuropeanOption::~EuropeanOption() {

}

SimulationResult EuropeanOption::callPayoff() {
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

SimulationResult EuropeanOption::putPayoff() {
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


SimulationResult EuropeanOption::callPayoffSerialCPU() {
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
        samples[i] = discoutCall(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
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

SimulationResult EuropeanOption::putPayoffSerialCPU() {
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
        samples[i] = discoutPut(getAsset()->getRiskFreeRate(), timeToMaturity, S_T, strikePrice);
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

float EuropeanOption::callPayoffBlackSholes(){

    float b = exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    float x1 = log(getAsset()->getSpotPrice() / (b * strikePrice)) + 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x1 = x1/(getAsset()->getVolatility()* (pow(timeToMaturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = z1 * getAsset()->getSpotPrice();

    float x2 = log(getAsset()->getSpotPrice() / (b * strikePrice)) - 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x2 = x2/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = b * strikePrice * z2;

    return z1 - z2;
}

float EuropeanOption::putPayoffBlackSholes(){

    float b = exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    float x1 = log( (b * strikePrice)/getAsset()->getSpotPrice()) + 0.5 * (pow(getAsset()->getVolatility(), 2) * getTimeToMaturity());
    x1 = x1/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = b * getAsset()->getSpotPrice() * z1;

    float x2 = log((b * strikePrice)/getAsset()->getSpotPrice()) - 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x2 = x2/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = strikePrice * z2;

    return z1 - z2;
}



