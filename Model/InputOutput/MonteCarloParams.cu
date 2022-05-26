//
// Created by marco on 19/05/22.
//

#include "MonteCarloParams.cuh"

#include <curand.h>


MonteCarloParams::MonteCarloParams(int nSimulations, curandRngType rngType, unsigned long long int seed) : nSimulations(
        nSimulations), rngType(rngType), seed(seed) {}

int MonteCarloParams::getNSimulations() const {
    return nSimulations;
}

void MonteCarloParams::setNSimulations(int nSimulations) {
    this->nSimulations = nSimulations;
}


curandRngType MonteCarloParams::getRngType() const {
    return rngType;
}

void MonteCarloParams::setRngType(curandRngType rngType) {
    this->rngType = rngType;
}

unsigned long long int MonteCarloParams::getSeed() const {
    return seed;
}

void MonteCarloParams::setSeed(unsigned long long int seed) {
    this->seed = seed;
}




