//
// Created by marco on 30/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTIONGPU_CUH
#define OPTIONPRICING_BINARYOPTIONGPU_CUH


#include "../../../InputOutput/MonteCarloParams.cuh"
#include "../../../InputOutput/GPUParams.cuh"
#include "../BinaryOption.cuh"

class BinaryOptionGPU : public BinaryOption{
protected:
    MonteCarloParams *monteCarloParams;

    GPUParams *gpuParams;
public:
    BinaryOptionGPU(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase,
                    MonteCarloParams *monteCarloParams, GPUParams *gpuParams);
    ~BinaryOptionGPU() = default;

    SimulationResult callPayoff();

    SimulationResult putPayoff();

};


#endif //OPTIONPRICING_BINARYOPTIONGPU_CUH
