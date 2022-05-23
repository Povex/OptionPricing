//
// Created by marco on 19/05/22.
//

#include "MonteCarloParams.cuh"

MonteCarloParams::MonteCarloParams(int nSimulations, int prngType) : nSimulations(nSimulations), PRNGType(prngType) {}

int MonteCarloParams::getNSimulations() const {
    return nSimulations;
}

void MonteCarloParams::setNSimulations(int nSimulations) {
    MonteCarloParams::nSimulations = nSimulations;
}

int MonteCarloParams::getPrngType() const {
    return PRNGType;
}

void MonteCarloParams::setPrngType(int prngType) {
    PRNGType = prngType;
}


