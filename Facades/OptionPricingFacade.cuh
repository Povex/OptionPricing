//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_OPTIONPRICINGFACADE_CUH
#define OPTIONPRICING_OPTIONPRICINGFACADE_CUH


#include "../Model/InputOutput/SimulationResult.cuh"

using namespace std;

class OptionPricingFacade {
public:
    vector<SimulationResult> executeEuropeanCall();

    SimulationResult executeEuropeanPut();

    SimulationResult executeAutoCallableCall();
};


#endif //OPTIONPRICING_OPTIONPRICINGFACADE_CUH
