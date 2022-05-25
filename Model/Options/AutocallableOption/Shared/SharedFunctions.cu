//
// Created by marco on 24/05/22.
//

#include "SharedFunctions.cuh"

__host__ __device__
void autoCallablePayoff(float spotPrice,
                        float riskFreeRate,
                        float volatility,
                        float rebase,
                        float *d_samples,
                        const float *d_normals,
                        const float *d_observationDates,
                        const float *d_barriers,
                        const float *d_payoffs,
                        int dateBarrierSize,
                        unsigned int i){
    bool barrier_hit = false;
    float S = spotPrice;
    int date_index = 0;
    float dt = d_observationDates[date_index];

    while (date_index <= dateBarrierSize - 1) {
        S = S * exp((riskFreeRate - (pow(volatility, 2) / 2)) * dt
                    + volatility* sqrt(dt) * d_normals[i + date_index]);

        if (S >= d_barriers[date_index]) { barrier_hit = true; break; }

        date_index++;
        dt = d_observationDates[date_index] - d_observationDates[date_index - 1];
    }

    if(!barrier_hit)
        d_samples[i] = exp(-riskFreeRate * d_observationDates[dateBarrierSize-1]) * rebase;
    else
        d_samples[i] = exp(-riskFreeRate * d_observationDates[date_index]) * d_payoffs[date_index];
}