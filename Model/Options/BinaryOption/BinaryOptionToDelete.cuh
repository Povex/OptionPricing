//
// Created by marco on 07/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTIONTODELETE_CUH
#define OPTIONPRICING_BINARYOPTIONTODELETE_CUH


#include "../../InputOutput/Asset.cuh"
#include "../../InputOutput/SimulationResult.cuh"

class BinaryOptionToDelete {

public:
    Asset asset;
    float strike_price; // In this case is equal to payoff at time_to_maturity
    float time_to_maturity; // [years]
    float payoff;

    BinaryOptionToDelete();
    BinaryOptionToDelete(Asset market, float strike_price, float time_to_maturity, float payoff);
    BinaryOptionToDelete(BinaryOptionToDelete &option);

    float actual_call_payoff_blackSholes();
    float actual_put_payoff_blackSholes();

    float call_payoff_montecarlo_gpu(const int n_simulations);
    float put_payoff_montecarlo_gpu(const int n_simulations);

};


#endif //OPTIONPRICING_BINARYOPTIONTODELETE_CUH
