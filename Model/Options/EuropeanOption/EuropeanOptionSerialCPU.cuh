//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH
#define OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH


#include "EuropeanOption.cuh"

class EuropeanOptionSerialCPU : public EuropeanOption{
protected:
    MonteCarloParams *monteCarloParams;

    GPUParams *gpuParams;
public:
    EuropeanOptionSerialCPU(Asset *asset, float strikePrice, float timeToMaturity, MonteCarloParams *monteCarloParams,
                            GPUParams *gpuParams);

    ~EuropeanOptionSerialCPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    GPUParams *getGpuParams() const;

    void setGpuParams(GPUParams *gpuParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_EUROPEANOPTIONSERIALCPU_CUH
