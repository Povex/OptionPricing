//
// Created by marco on 02/05/22.
//

#include "SimulationResult.cuh"

SimulationResult::SimulationResult(){}

SimulationResult::SimulationResult(float value, const std::vector<float> &confidence, float stdError, float timeElapsed)
        : Result(value), confidence(confidence), stdError(stdError), timeElapsed(timeElapsed) {}


const std::vector<float> &SimulationResult::getConfidence() const {
    return confidence;
}

void SimulationResult::setConfidence(const std::vector<float> confidence) {
    this->confidence = confidence;
}

float SimulationResult::getStdError() const {
    return stdError;
}

void SimulationResult::setStdError(float stdError) {
    this->stdError = stdError;
}

float SimulationResult::getTimeElapsed() const {
    return timeElapsed;
}

void SimulationResult::setTimeElapsed(float timeElapsed) {
    SimulationResult::timeElapsed = timeElapsed;
}

std::ostream &operator<<(std::ostream &os, const SimulationResult &result) {
    os << result.getValue() << " confidence: [" << result.confidence[0] << ", " << result.confidence[1] << "]"<< " stdError: "
       << result.stdError << " timeElapsed: " << result.timeElapsed;
    return os;
}

SimulationResult::SimulationResult(float value) : Result(value) {}


