//
// Created by marco on 31/05/22.
//

#ifndef OPTIONPRICING_SHAREDBINARYOPTION_CUH
#define OPTIONPRICING_SHAREDBINARYOPTION_CUH

__host__ __device__
void binaryOptionSample(float spotPrice,
                      float riskFreeRate,
                      float volatility,
                      float timeToMaturity,
                      float barrier,
                      float *normals,
                      float *samples,
                      const int i);

#endif //OPTIONPRICING_SHAREDBINARYOPTION_CUH
