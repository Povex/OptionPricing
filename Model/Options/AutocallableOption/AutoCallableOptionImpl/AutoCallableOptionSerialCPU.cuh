//
// Created by marco on 24/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTIONSERIALCPU_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTIONSERIALCPU_CUH


#include "../../../InputOutput/MonteCarloParams.cuh"
#include "../../../InputOutput/SimulationResult.cuh"
#include "../AutoCallableOption.cuh"

class AutoCallableOptionSerialCPU : public AutoCallableOption{
protected:
    MonteCarloParams *monteCarloParams;
public:
    AutoCallableOptionSerialCPU(Asset *asset, float rebase, const std::vector<float> &observationDates,
                          const std::vector<float> &barriers, const std::vector<float> &payoffs,
                          MonteCarloParams *monteCarloParams);

    ~AutoCallableOptionSerialCPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_AUTOCALLABLEOPTIONSERIALCPU_CUH
