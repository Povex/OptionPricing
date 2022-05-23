//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTIONANALYTICAL_CUH
#define OPTIONPRICING_EUROPEANOPTIONANALYTICAL_CUH


#include "EuropeanOption.cuh"

class EuropeanOptionAnalytical : public EuropeanOption {
public:
    EuropeanOptionAnalytical(Asset *asset, float strikePrice, float timeToMaturity);
    ~EuropeanOptionAnalytical() = default;

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;
};


#endif //OPTIONPRICING_EUROPEANOPTIONANALYTICAL_CUH
