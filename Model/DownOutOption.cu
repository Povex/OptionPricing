//
// Created by marco on 03/05/22.
//

#include "DownOutOption.cuh"

#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <curand_kernel.h>

#include <random>
#include "Asset.cuh"

using namespace std;

__global__ void K_call_payoff(Asset asset, float time_to_maturity, float strike_price, float barrier, float *d_samples, float *d_normals, int n_intervals, float dt, const unsigned int n_paths){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n_paths) return;

    bool barrier_hit = false;
    float S = asset.spot_price;

    for(int j=0; j<n_intervals; j++){
        // Approssimazione di eulero
        //S = S * (1 + (market.risk_free_rate * dt) + (market.volatility * sqrt(dt) * d_normals[i * n_intervals + j] ));

        // Formula esatta
        S = S * exp((asset.risk_free_rate - (pow(asset.volatility, 2) / 2)) * dt + asset.volatility * sqrt(dt) * d_normals[i * n_intervals + j]);

        if(S <= barrier){
            barrier_hit = true;
            break;
        }
    }

    float C_i = 0.0f;
    if(!barrier_hit)
        C_i = exp(-asset.risk_free_rate * time_to_maturity) * fmaxf(0, S - strike_price);
    // else rebase
    d_samples[i] = C_i;
}

DownOutOption::DownOutOption(){} // Definire meglio questo costruttore

DownOutOption::DownOutOption(Asset asset, float strike_price, float time_to_maturity, float barrier){
    this->asset = asset;
    this->strike_price = strike_price;
    this->time_to_maturity = time_to_maturity;
    this->barrier = barrier; // Nelle Down Deve essere < asset price implementare il check
    this->n_intervals = 365;
    this->dt = 1.0/(float)n_intervals;
    this->rebase = 0.0f;
}

SimulationResult DownOutOption::call_payoff(){
    const int N_THREADS = 256;
    const int N_PATHS = 100000;

    size_t size = sizeof(float) * N_PATHS;
    float *h_samples = (float *) malloc(size);

    float *d_samples = NULL;
    cudaMalloc((void **)&d_samples, size);


    float *d_normals = NULL;
    cudaMalloc((void **)&d_normals, size * n_intervals);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormal(generator, d_normals, N_PATHS * n_intervals, 0.0f, 1.0f);
    //cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    K_call_payoff<<<(N_PATHS + N_THREADS - 1)/N_THREADS, N_THREADS>>>(asset, time_to_maturity, strike_price,barrier, d_samples, d_normals, n_intervals, dt, N_PATHS);

    cudaDeviceSynchronize();

    // copy results from device to host
    cudaMemcpy(h_samples, d_samples, size, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);

    SimulationResult result(h_samples, N_PATHS, 0);

    free(h_samples);

    cudaFree(d_samples);
    cudaFree(d_normals);

    return result;
}

SimulationResult DownOutOption::call_payoff_cpu(){
    mt19937 gen(static_cast<long unsigned int>(time(0)));
    normal_distribution<double> distribution(0.0f, 1.0f);

    const int N_PATHS = 100000;

    bool barrier_hit = false;

    float S = asset.spot_price;
    float z = 0;

    float *C = (float *)malloc(sizeof(float) * N_PATHS);
    for(int i=0; i<N_PATHS; i++){
        barrier_hit = false;
        S = asset.spot_price;
        for(int j=0; j< n_intervals; j++){
            z = distribution(gen);
            //S = S * (1 + (market.risk_free_rate * dt) + (market.volatility * sqrt(dt) * z));
            S = S * exp((asset.risk_free_rate - (pow(asset.volatility, 2) / 2)) * dt + asset.volatility * sqrt(dt) * z);

            if(S <= barrier){
                barrier_hit = true;
                break;
            }
        }

        float C_i = 0.0f;
        if(!barrier_hit)
            C_i = exp(-asset.risk_free_rate * time_to_maturity) * fmax(0, S - strike_price);
        // else rebase
        C[i] = C_i;
    }

    SimulationResult result(C, N_PATHS, 0);
    return result;
}