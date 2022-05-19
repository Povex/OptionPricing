//
// Created by marco on 05/05/22.
//

#include <iostream>

#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>

#include "AutocallableOption.cuh"

__global__ void K_call_payoff(AutoCallableOption option, float *d_samples, float *d_normals, int n_paths,
                              float *d_observationDates, float *d_barriers, float *d_payoffs, int dateBarrierSize){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n_paths) return;

    float barrier_hit = false;
    float S = option.asset.spot_price;
    int date_index = 0;
    float dt = d_observationDates[date_index];

    while (date_index <= dateBarrierSize - 1) {
        S = S * exp((option.asset.risk_free_rate - (pow(option.asset.volatility, 2) / 2)) * dt
                + option.asset.volatility * sqrt(dt) * d_normals[i + date_index]);

        if (S >= d_barriers[date_index]) { barrier_hit = true; break; }

        date_index++;
        dt = d_observationDates[date_index] - d_observationDates[date_index - 1];
    }

    if(!barrier_hit)
        d_samples[i] = exp(-option.asset.risk_free_rate * d_observationDates[dateBarrierSize-1]) * option.rebase;
    else
        d_samples[i] = exp(-option.asset.risk_free_rate * d_observationDates[date_index]) * d_payoffs[date_index];
}

AutoCallableOption::AutoCallableOption(){} // Definire meglio questo costruttore

AutoCallableOption::AutoCallableOption(Asset asset, float rebase, vector<float> observationDates,
                                       vector<float> barriers,  vector<float> payoffs){
    this->asset = asset;
    this->observationDates = observationDates;
    this->barriers = barriers;
    this->payoffs = payoffs;
    this->rebase = rebase;
}

AutoCallableOption::AutoCallableOption(AutoCallableOption &option){
    this->asset = option.asset;
    this->n_intervals = 3 * 365; // must take the last observation date
    this->dt = 1.0/(float) 365; // day increment
    this->rebase = option.rebase;
    this->dateBarrier = option.dateBarrier;
}

SimulationResult AutoCallableOption::call_payoff_montecarlo_cpu(){
    mt19937 gen(static_cast<long unsigned int>(time(0)));
    normal_distribution<double> distribution(0.0f, 1.0f);

    const int N_PATHS = 10000000;
    bool barrier_hit = false;

    float S = asset.spot_price;
    float z = 0;

    float *C = (float *)malloc(sizeof(float) * N_PATHS);

    int date_index = 0;
    float dt = 0.0f;

    for(int i=0; i<N_PATHS; i++) {
        barrier_hit = false;
        S = asset.spot_price;
        date_index = 0;
        dt = observationDates[date_index];
        while (date_index <= observationDates.size() - 1) {
            z = distribution(gen);
            S = S * exp((asset.risk_free_rate - (pow(asset.volatility, 2) / 2)) * dt + asset.volatility * sqrt(dt) * z);

            if (S >= barriers[date_index]) {
                    barrier_hit = true;
                    break;
            }

            date_index++;
            dt = observationDates[date_index] - observationDates[date_index - 1];
        }

        if(!barrier_hit) C[i] = exp(-asset.risk_free_rate * observationDates[observationDates.size()-1]) * rebase;
        else C[i] = exp(-asset.risk_free_rate * observationDates[date_index]) * payoffs[date_index];
    }

    SimulationResult result(C, N_PATHS, 0);
    return result;
}

SimulationResult  AutoCallableOption::call_payoff_montecarlo_gpu(){
    const int N_THREADS = 256;
    const int N_PATHS = 10000000;

    size_t size = sizeof(float) * N_PATHS;
    float *h_samples = (float *) malloc(size);

    float *d_samples = nullptr;
    cudaMalloc((void **)&d_samples, size);

    float *d_normals = nullptr;
    cudaMalloc((void **)&d_normals, size * observationDates.size());

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormal(generator, d_normals, N_PATHS * observationDates.size(), 0.0f, 1.0f);
    //cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

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

    AutoCallableOption option(*this);
    K_call_payoff<<<(N_PATHS + N_THREADS - 1)/N_THREADS, N_THREADS>>>(option, d_samples, d_normals, N_PATHS, d_observationDates, d_barriers, d_payoffs, observationDates.size());

    cudaDeviceSynchronize();

    // copy results from device to host
    cudaMemcpy(h_samples, d_samples, size, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);

    SimulationResult result(h_samples, N_PATHS, 0);

    free(h_samples);

    cudaFree(d_samples);
    cudaFree(d_normals);
    cudaFree(d_observationDates);
    cudaFree(d_samples);
    cudaFree(d_barriers);

    return result;
}