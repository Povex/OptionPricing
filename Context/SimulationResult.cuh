//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_SIMULATIONRESULT_CUH
#define OPTIONPRICING_SIMULATIONRESULT_CUH

#include <ostream>
#include <vector>
#include "Result.cuh"

class SimulationResult : public Result{

private:
    std::vector<float> confidence;
    float stdError;
    float timeElapsed;

public:
    SimulationResult();
    SimulationResult(float value, const std::vector<float> &confidence, float stdError, float timeElapsed);

    const std::vector<float> &getConfidence() const;

    void setConfidence(const std::vector<float> confidence);

    float getStdError() const;

    void setStdError(float stdError);

    float getTimeElapsed() const;

    void setTimeElapsed(float timeElapsed);

    friend std::ostream &operator<<(std::ostream &os, const SimulationResult &result);

};

#endif //OPTIONPRICING_SIMULATIONRESULT_CUH
