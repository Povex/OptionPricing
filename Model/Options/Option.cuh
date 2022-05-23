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

public:
    Option();

    Option(Asset *asset);

    ~Option() = default;

    Asset *getAsset() const;

    void setAsset(Asset *asset);

    virtual SimulationResult callPayoff() = 0;

    virtual SimulationResult putPayoff() = 0;

};

#endif //OPTIONPRICING_OPTION_CUH
