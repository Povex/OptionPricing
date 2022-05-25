//
// Created by marco on 24/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTIONCPU_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTIONCPU_CUH


#include "../../../InputOutput/MonteCarloParams.cuh"
#include "../../../InputOutput/SimulationResult.cuh"
#include "../AutoCallableOption.cuh"

class AutoCallableOptionCPU : public AutoCallableOption{
protected:
    MonteCarloParams *monteCarloParams;
public:
    AutoCallableOptionCPU(Asset *asset, float rebase, const std::vector<float> &observationDates,
                          const std::vector<float> &barriers, const std::vector<float> &payoffs,
                          MonteCarloParams *monteCarloParams);

    ~AutoCallableOptionCPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_AUTOCALLABLEOPTIONCPU_CUH
