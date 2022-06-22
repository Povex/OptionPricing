//
// Created by marco on 21/06/22.
//

#include "StatisticsCPU.cuh"

#include <numeric>

void StatisticsCPU::calcMean() {
    double temp = 0.0;
    #pragma omp parallel for reduction (+:temp)
    for(int i=0; i<n; i++) {
        temp += samples[i];
    }

    double mean = temp /n;
    setMean(mean);
}

void StatisticsCPU::calcCI() {
    // Calculate confidence interval 95% serially
    // Assumes that mean is already calculated
    double temp = 0.0;
    #pragma omp parallel for reduction (+:temp)
    for(int i=0; i<n; i++){
        temp += pow(samples[i] - mean, 2);
    }

    double stdDev = sqrt(temp/(n-1));
    double stdError = stdDev/(sqrt(n));
    setStdDev(stdDev);
    setStdError(stdError);

    // Calculate confidence interval 95%
    std::vector<float> confidence(2);
    confidence[0] = mean - 1.96 * stdError;
    confidence[1] = mean + 1.96 * stdError;
    setConfidence(confidence);
}

StatisticsCPU::StatisticsCPU(float *samples, unsigned int n) : samples(samples), n(n) {}

