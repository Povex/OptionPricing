//
// Created by marco on 31/05/22.
//

#include "SharedBinaryOption.cuh"
#include "../../Shared/SharedFunctions.cuh"

__host__ __device__
void binaryOptionSample(float spotPrice,
                      float riskFreeRate,
                      float volatility,
                      float timeToMaturity,
                      float barrier,
                      float *normals,
                      float *samples,
                      const int i) {

    float stockT = generateS_T(spotPrice,
                         riskFreeRate,
                         volatility,
                         timeToMaturity,
                         normals[i]);

    samples[i] = stockT >= barrier ? 1 : 0;
}