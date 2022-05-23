//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_SHAREDFUNCTIONS_CUH
#define OPTIONPRICING_SHAREDFUNCTIONS_CUH

__host__ __device__
float discountCall(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice);

__host__ __device__
float discountPut(float riskFreeRate, float timeToMaturity, float S_T, float strikePrice);

__host__ __device__
float generateS_T(float spotPrice,
                  float riskFreeRate,
                  float volatility,
                  float timeToMaturity,
                  float z);

#endif //OPTIONPRICING_SHAREDFUNCTIONS_CUH
