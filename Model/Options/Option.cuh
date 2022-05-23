//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_OPTION_CUH
#define OPTIONPRICING_OPTION_CUH


#include "../InputOutput/Asset.cuh"
#include "../InputOutput/GPUParams.cuh"
#include "../InputOutput/MonteCarloParams.cuh"
#include "../InputOutput/SimulationResult.cuh"

class Option {
private:
    Asset *asset;

    GPUParams *gpuParams;

    MonteCarloParams *monteCarloParams;

public:
    Option();

    Option(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams);

     ~Option() = default;

    Asset *getAsset() const;

    void setAsset(Asset *asset);

    GPUParams *getGpuParams() const;

    void setGpuParams(GPUParams *gpuParams);

    MonteCarloParams *getMonteCarloParams() const;

    void setMonteCarloParams(MonteCarloParams *monteCarloParams);

    virtual SimulationResult callPayoff() = 0;

    virtual SimulationResult putPayoff() = 0;

};

#endif //OPTIONPRICING_OPTION_CUH