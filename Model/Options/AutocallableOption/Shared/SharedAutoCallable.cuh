//
// Created by marco on 24/05/22.
//

#ifndef OPTIONPRICING_SHAREDAUTOCALLABLE_CUH
#define OPTIONPRICING_SHAREDAUTOCALLABLE_CUH

__host__ __device__
void autoCallablePayoff(float spotPrice, float riskFreeRate, float volatility, float rebase, float *d_samples,
                        const float *d_normals, const float *d_observationDates, const float *d_barriers,
                        const float *d_payoffs, int dateBarrierSize, unsigned int i, const int n_path);

#endif //OPTIONPRICING_SHAREDAUTOCALLABLE_CUH
