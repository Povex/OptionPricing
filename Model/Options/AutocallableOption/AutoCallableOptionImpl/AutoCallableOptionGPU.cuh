//
// Created by marco on 24/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTIONGPU_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTIONGPU_CUH


#include "../AutoCallableOption.cuh"

class AutoCallableOptionGPU : public AutoCallableOption{

protected:
    MonteCarloParams *monteCarloParams;

    GPUParams *gpuParams;

public:
    AutoCallableOptionGPU(Asset *asset, float rebase, const std::vector<float> &observationDates,
                          const std::vector<float> &barriers, const std::vector<float> &payoffs,
                          MonteCarloParams *monteCarloParams, GPUParams *gpuParams);

    ~AutoCallableOptionGPU() = default;

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    GPUParams *getGpuParams() const;

    void setGpuParams(GPUParams *gpuParams);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;

};


#endif //OPTIONPRICING_AUTOCALLABLEOPTIONGPU_CUH
