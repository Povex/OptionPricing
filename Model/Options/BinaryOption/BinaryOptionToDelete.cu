//
// Created by marco on 07/05/22.
// Reference: https://www.codearmo.com/python-tutorial/binary-options-and-implied-distributions
//

#include "BinaryOptionToDelete.cuh"


#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <random>

using namespace std;
/*
__global__ void is_over_barrier(BinaryOptionToDelete option, float *d_samples, float *d_normals, const unsigned int n_paths){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n_paths) return;

    float stock_T = option.asset.spot_price * exp((option.asset.risk_free_rate - 0.5 * pow(option.asset.volatility, 2))
                                                  * option.time_to_maturity + option.asset.volatility * sqrt(option.time_to_maturity) * d_normals[i]);

    d_samples[i] = stock_T >= option.strike_price ? 1 : 0;
}

BinaryOptionToDelete::BinaryOptionToDelete(){} // Definire meglio questo costruttore

BinaryOptionToDelete::BinaryOptionToDelete(Asset asset, float strike_price, float time_to_maturity, float payoff){
    this->asset = asset;
    this->strike_price = strike_price;
    this->time_to_maturity = time_to_maturity;
    this->payoff = payoff;
}

BinaryOptionToDelete::BinaryOptionToDelete(BinaryOptionToDelete &option){
    this->asset = option.asset;
    this->strike_price = option.strike_price;
    this->time_to_maturity = option.time_to_maturity;
    this->payoff = option.payoff;
}

float BinaryOptionToDelete::call_payoff_montecarlo_gpu(const int n_simulations){
    const int N_THREADS = 256;
    const int N_PATHS = n_simulations;

    size_t size = sizeof(float) * N_PATHS;
    float *h_samples = (float *) malloc(size);

    float *d_samples = NULL;
    cudaMalloc((void **)&d_samples, size);


    float *d_normals = NULL;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormal(generator, d_normals, N_PATHS, 0.0f, 1.0f);
    //cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    BinaryOptionToDelete option(*this);
    is_over_barrier<<<(N_PATHS + N_THREADS - 1)/N_THREADS, N_THREADS>>>(option, d_samples, d_normals, N_PATHS);

    cudaDeviceSynchronize();

    // copy results from device to host
    cudaMemcpy(h_samples, d_samples, size, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);

    float tmp = 0.0f;
    for(int i=0; i<n_simulations; i++){
        tmp += h_samples[i];
    }
    float prob_greater_barrier = tmp / (float) n_simulations;
    float call_value = prob_greater_barrier * option.payoff * exp(-option.asset.risk_free_rate * option.time_to_maturity);


    free(h_samples);

    cudaFree(d_samples);
    cudaFree(d_normals);

    return call_value;
}

float BinaryOptionToDelete::put_payoff_montecarlo_gpu(const int n_simulations){
    const int N_THREADS = 256;
    const int N_PATHS = n_simulations;

    size_t size = sizeof(float) * N_PATHS;
    float *h_samples = (float *) malloc(size);

    float *d_samples = NULL;
    cudaMalloc((void **)&d_samples, size);


    float *d_normals = NULL;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormal(generator, d_normals, N_PATHS, 0.0f, 1.0f);
    //cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost);

    BinaryOptionToDelete option(*this);
    is_over_barrier<<<(N_PATHS + N_THREADS - 1)/N_THREADS, N_THREADS>>>(option, d_samples, d_normals, N_PATHS);

    cudaDeviceSynchronize();

    // copy results from device to host
    cudaMemcpy(h_samples, d_samples, size, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);

    float tmp = 0.0f;
    for(int i=0; i<n_simulations; i++){
        tmp += h_samples[i];
    }
    float prob_under_barrier = (n_simulations - tmp) / (float) n_simulations;
    float put_value = prob_under_barrier * payoff * exp(-option.asset.risk_free_rate * option.time_to_maturity);


    free(h_samples);

    cudaFree(d_samples);
    cudaFree(d_normals);

    return put_value;
}

float normalCDF(float value)
{
    return  0.5 * erfcf(-value * M_SQRT1_2);
}

float BinaryOptionToDelete::actual_call_payoff_blackSholes() {

    float d1 = log(asset.spot_price / strike_price) + (asset.risk_free_rate + 0.5 * pow(asset.volatility, 2)) * time_to_maturity;
    d1 = d1 / (asset.volatility * sqrtf(time_to_maturity));

    float d2 = d1 - asset.volatility * sqrt(time_to_maturity);

    float in_money_probability = normalCDF(d2) * exp(asset.risk_free_rate * time_to_maturity);
    float call_value = in_money_probability * payoff * exp(-asset.risk_free_rate * time_to_maturity);
    return call_value;
}

float BinaryOptionToDelete::actual_put_payoff_blackSholes() {

    float d1 = log(asset.spot_price / strike_price) + (asset.risk_free_rate + 0.5 * pow(asset.volatility, 2)) * time_to_maturity;
    d1 = d1 / (asset.volatility * sqrtf(time_to_maturity));

    float d2 = d1 - asset.volatility * sqrt(time_to_maturity);

    float in_money_probability = normalCDF(-d2) * exp(asset.risk_free_rate * time_to_maturity);
    float put_value = in_money_probability * payoff * exp(-asset.risk_free_rate * time_to_maturity);
    return put_value;
}
*/


