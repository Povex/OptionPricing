//
// Created by marco on 21/06/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTIONCPU_CUH
#define OPTIONPRICING_EUROPEANOPTIONCPU_CUH


#include "../EuropeanOption.cuh"

class EuropeanOptionCPU : public EuropeanOption {
protected:
    MonteCarloParams *monteCarloParams;
public:
    EuropeanOptionCPU(Asset *asset, float strikePrice, float timeToMaturity, MonteCarloParams *monteCarloParams);

    ~EuropeanOptionCPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_EUROPEANOPTIONCPU_CUH
