//
// Created by marco on 23/05/22.
//

#include "StatisticUtils.cuh"

float StatisticUtils::getStdError() const {
    return stdError;
}

void StatisticUtils::setStdError(float stdError) {
    this->stdError = stdError;
}

float StatisticUtils::getStdDev() const {
    return stdDev;
}

void StatisticUtils::setStdDev(float stdDev) {
    this->stdDev = stdDev;
}

const std::vector<float> &StatisticUtils::getConfidence() const {
    return confidence;
}

void StatisticUtils::setConfidence(const std::vector<float> &confidence) {
    this->confidence = confidence;
}

float StatisticUtils::getMean() const {
    return mean;
}

void StatisticUtils::setMean(float mean) {
    this->mean = mean;
}

StatisticUtils::StatisticUtils(float stdError, float stdDev, const std::vector<float> &confidence, float mean)
        : stdError(stdError), stdDev(stdDev), confidence(confidence), mean(mean) {}

StatisticUtils::StatisticUtils() {
}





