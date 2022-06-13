//
// Created by marco on 31/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTIONCPU_CUH
#define OPTIONPRICING_BINARYOPTIONCPU_CUH


#include "../BinaryOption.cuh"

class BinaryOptionCPU : public BinaryOption{
protected:
    MonteCarloParams *monteCarloParams;

public:
    BinaryOptionCPU(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase,
                    MonteCarloParams *monteCarloParams);

    ~BinaryOptionCPU() = default;

    SimulationResult callPayoff();

    SimulationResult putPayoff();

};


#endif //OPTIONPRICING_BINARYOPTIONCPU_CUH
