//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH
#define OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH


#include "../EuropeanOption.cuh"

class EuropeanOptionSerialCPU : public EuropeanOption{
protected:
    MonteCarloParams *monteCarloParams;

public:
    EuropeanOptionSerialCPU(Asset *asset, float strikePrice, float timeToMaturity, MonteCarloParams *monteCarloParams);

    ~EuropeanOptionSerialCPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH
