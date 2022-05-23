//
// Created by marco on 07/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTION_CUH
#define OPTIONPRICING_BINARYOPTION_CUH


#include "../../Context/Asset.cuh"
#include "../../Context/SimulationResult.cuh"

class BinaryOption {

public:
    Asset asset;
    float strike_price; // In this case is equal to payoff at time_to_maturity
    float time_to_maturity; // [years]
    float payoff;

    BinaryOption();
    BinaryOption(Asset market, float strike_price, float time_to_maturity, float payoff);
    BinaryOption(BinaryOption &option);

    float actual_call_payoff_blackSholes();
    float actual_put_payoff_blackSholes();

    float call_payoff_montecarlo_gpu(const int n_simulations);
    float put_payoff_montecarlo_gpu(const int n_simulations);

};


#endif //OPTIONPRICING_BINARYOPTION_CUH
