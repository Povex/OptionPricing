//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_SIMULATIONRESULT_CUH
#define OPTIONPRICING_SIMULATIONRESULT_CUH

class SimulationResult {

private:
    float value;
    float confidenceInterval95[2];
    float stdError;
    float timeElapsed;

public:
    SimulationResult();
    SimulationResult(float *samples, int n, float timeElapsed);
    float getValue();
    float* getConfidenceInterval();
    float getStdError();
    float getTimeElapsed();
};

#endif //OPTIONPRICING_SIMULATIONRESULT_CUH
