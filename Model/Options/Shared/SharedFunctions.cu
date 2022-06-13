//
// Created by marco on 23/05/22.
//

#include "SharedFunctions.cuh"

__host__ __device__
float discountCall(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice){
    return expf(-riskFreeRate * timeToMaturity) * fmaxf(0, S_T - strikePrice);
}

__host__ __device__
float discountPut(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice){
    return expf(-riskFreeRate * timeToMaturity) * fmaxf(0, strikePrice - S_T);
}

__host__ __device__
float generateS_T(float spotPrice,
                  float riskFreeRate,
                  float volatility,
                  float dt,
                  float z){

    return spotPrice * expf((riskFreeRate - 0.5f * powf(volatility, 2))
                            * dt + volatility * sqrtf(dt) * z);
}

__host__ __device__
float normalCDF(float value){
    return  0.5 * erfc(-value * M_SQRT1_2);
}