//
// Created by marco on 02/05/22.
//

#include "EuropeanOption.cuh"

#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <curand_kernel.h>

#include <random>

using namespace std;

__global__ void K_call_payoff(EuropeanOption option, float *d_samples, float *d_normals, const unsigned int n_paths){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n_paths) return;

    float S_T = option.market.spot_price * exp((option.market.risk_free_rate - 0.5 * pow(option.market.volatility, 2)) * option.time_to_maturity + option.market.volatility * sqrt(option.time_to_maturity) * d_normals[i]);
    float C = exp(-option.market.risk_free_rate * option.time_to_maturity) * fmaxf(0, S_T - option.strike_price); // discount by riskfree rate payoff
    d_samples[i] = C;
}

EuropeanOption::EuropeanOption(){} // Definire meglio questo costruttore

EuropeanOption::EuropeanOption(Asset market, float strike_price, float time_to_maturity){
    this->market = market;
    this->strike_price = strike_price;
    this->time_to_maturity = time_to_maturity;
}

EuropeanOption::EuropeanOption(EuropeanOption &option){
    this->market = option.market;
    this->strike_price = option.strike_price;
    this->time_to_maturity = option.time_to_maturity;
}


float EuropeanOption::stock_T(){
    default_random_engine eng{static_cast<long unsigned int>(time(0)) };
    normal_distribution<double> distribution(0.0f, 1.0f);

    float stock_T = 0.0f;
    float z = distribution(eng); // sample from std normal distribution

    stock_T = market.spot_price * exp((market.risk_free_rate - 0.5 * pow(market.volatility, 2))
                                      * time_to_maturity + market.volatility * sqrt(time_to_maturity) * z);

    return stock_T;
}

float EuropeanOption::call_payoff_blackSholes(){

    float b = exp(-market.risk_free_rate * time_to_maturity);

    float x1 = log(market.spot_price / (b * strike_price)) + 0.5 * (pow(market.volatility, 2) * time_to_maturity);
    x1 = x1/(market.volatility * (pow(time_to_maturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = z1 * market.spot_price;

    float x2 = log(market.spot_price / (b * strike_price)) - 0.5 * (pow(market.volatility, 2) * time_to_maturity);
    x2 = x2/(market.volatility * (pow(time_to_maturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = b * strike_price * z2;

    return z1 - z2;
}


SimulationResult EuropeanOption::call_payoff_montecarlo(const unsigned int n_simulations){
    const int N_THREADS = 256;

    size_t size = sizeof(float) * n_simulations;
    float *h_samples = (float *) malloc(size);

    float *d_samples = NULL;
    cudaMalloc((void **)&d_samples, size);

    float *h_normals = (float *) malloc(size);
    float *d_normals = NULL;
    cudaMalloc((void **)&d_normals, size);

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormal(generator, d_normals, n_simulations, 0.0f, 1.0f);

    EuropeanOption option(*this); // *this -> &option

    K_call_payoff<<<(n_simulations + N_THREADS - 1)/N_THREADS, N_THREADS>>>(option, d_samples, d_normals, n_simulations);

    cudaDeviceSynchronize();

    // copy results from device to host
    cudaMemcpy(h_samples, d_samples, size, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);

    SimulationResult result(h_samples, n_simulations, 0);

    free(h_samples); free(h_normals);
    cudaFree(d_samples); cudaFree(d_normals);

    return result;
}

SimulationResult EuropeanOption::call_payoff_montecarlo_cpu(const unsigned int n_simulations){
    default_random_engine eng{static_cast<long unsigned int>(time(0)) };
    normal_distribution<double> distribution(0.0f, 1.0f);

    float *samples = (float *)malloc(n_simulations * sizeof (float));

    float S_T = 0.0f;
    float C = 0.0f;
    for(int i=0; i<n_simulations;i++){
        S_T = market.spot_price * exp((market.risk_free_rate - 0.5 * pow(market.volatility, 2)) * time_to_maturity + market.volatility * sqrt(time_to_maturity) * distribution(eng));
        C = exp(-market.risk_free_rate * time_to_maturity) * fmax(0, S_T - strike_price); // discount by riskfree rate payoff
        samples[i] = C;
    }

    SimulationResult result(samples, n_simulations, 0);
    return result;
}


float EuropeanOption::put_payoff_blackSholes(){

    float b = exp(-market.risk_free_rate * time_to_maturity);

    float x1 = log( (b*strike_price)/market.spot_price) + 0.5 * (pow(market.volatility, 2) * time_to_maturity);
    x1 = x1/(market.volatility * (pow(time_to_maturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = b * market.spot_price * z1;

    float x2 = log((b*strike_price)/market.spot_price) - 0.5 * (pow(market.volatility, 2) * time_to_maturity);
    x2 = x2/(market.volatility * (pow(time_to_maturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = strike_price * z2;

    return z1 - z2;
}

