//
// Created by marco on 30/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTIONANALYTICAL_CUH
#define OPTIONPRICING_BINARYOPTIONANALYTICAL_CUH


#include "../BinaryOption.cuh"

class BinaryOptionAnalytical : public BinaryOption {
public:
    BinaryOptionAnalytical(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase);

    SimulationResult callPayoff();

    SimulationResult putPayoff();
};


#endif //OPTIONPRICING_BINARYOPTIONANALYTICAL_CUH
