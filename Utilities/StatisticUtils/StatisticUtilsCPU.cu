//
// Created by marco on 23/05/22.
//

#include <numeric>
#include <iostream>
#include "StatisticUtilsCPU.cuh"

void StatisticUtilsCPU::calcMean() {
    float temp = std::accumulate(samples, samples + n - 1, 0.0f);
    float mean = temp / float(n);
    setMean(mean);
}

void StatisticUtilsCPU::calcCI() {
    // Calculate confidence interval 95% serially
    // Assumes that mean is already calculated
    float temp = 0.0f;
    for(int i=0; i<n; i++){
        temp += powf(samples[i] - mean, 2);
    }
    float stdDev = sqrtf(temp/float(n-1));
    float stdError = stdDev/(sqrt(n));
    setStdDev(stdDev);
    setStdError(stdError);

    // Calculate confidence interval 95%
    std::vector<float> confidence(2);
    confidence[0] = mean - 1.96 * stdError;
    confidence[1] = mean + 1.96 * stdError;
    setConfidence(confidence);
}

StatisticUtilsCPU::StatisticUtilsCPU(float *samples, unsigned int n) : samples(samples), n(n) {}






