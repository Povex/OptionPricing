//
// Created by marco on 02/05/22.
//

#include "SimulationResult.cuh"

SimulationResult::SimulationResult(){}

float mean(float *samples, int n){
    float temp_value=0.0;
    for(int i=0; i<n; i++) {
        temp_value += samples[i];
    }
    temp_value /= (float) n;

    return temp_value;
}

SimulationResult::SimulationResult(float *samples, int n, float timeElapsed){

    // Mean
    this->value = mean(samples, n);

    // Confidence interval
    float tmp = 0.0f;
    for(int i=0; i<n; i++){
        tmp += pow(samples[i] - this->value, 2);
    }
    float s = sqrt(1/((float)n-1) * tmp);
    float stdError = s/(sqrt(n));
    confidenceInterval95[0] = this->value - 1.96 * stdError;
    confidenceInterval95[1] = this->value + 1.96 * stdError;

    // Standard error
    this->stdError = stdError;
    this->timeElapsed = timeElapsed;
}

float SimulationResult::getValue(){
    return this->value;
}

float* SimulationResult::getConfidenceInterval(){
    return this->confidenceInterval95;
}

float SimulationResult::getStdError(){
    return this->stdError;
}

float SimulationResult::getTimeElapsed(){
    return this->timeElapsed;
}
