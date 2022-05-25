//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTIONGPU_CUH
#define OPTIONPRICING_EUROPEANOPTIONGPU_CUH

#include "../EuropeanOption.cuh"

class EuropeanOptionGPU : public EuropeanOption{
protected:
    MonteCarloParams *monteCarloParams;

    GPUParams *gpuParams;
public:
    EuropeanOptionGPU(Asset *asset, float strikePrice, float timeToMaturity, MonteCarloParams *monteCarloParams,
                      GPUParams *gpuParams);

    ~EuropeanOptionGPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    GPUParams *getGpuParams() const;

    void setGpuParams(GPUParams *gpuParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;

};


#endif //OPTIONPRICING_EUROPEANOPTIONGPU_CUH
