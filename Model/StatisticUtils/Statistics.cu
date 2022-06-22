//
// Created by marco on 23/05/22.
//

#include "Statistics.cuh"

float Statistics::getStdError() const {
    return stdError;
}

void Statistics::setStdError(float stdError) {
    this->stdError = stdError;
}

float Statistics::getStdDev() const {
    return stdDev;
}

void Statistics::setStdDev(float stdDev) {
    this->stdDev = stdDev;
}

const std::vector<float> &Statistics::getConfidence() const {
    return confidence;
}

void Statistics::setConfidence(const std::vector<float> &confidence) {
    this->confidence = confidence;
}

float Statistics::getMean() const {
    return mean;
}

void Statistics::setMean(float mean) {
    this->mean = mean;
}

Statistics::Statistics(float stdError, float stdDev, const std::vector<float> &confidence, float mean)
        : stdError(stdError), stdDev(stdDev), confidence(confidence), mean(mean) {}

Statistics::Statistics() {
}





