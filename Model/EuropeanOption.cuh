//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTION_CUH
#define OPTIONPRICING_EUROPEANOPTION_CUH


#include "Asset.cuh"
#include "SimulationResult.cuh"

class EuropeanOption {

public:
    Asset market;
    float strike_price;
    float time_to_maturity;

    EuropeanOption();
    EuropeanOption(Asset market, float strike_price, float time_to_maturity);
    EuropeanOption(EuropeanOption &option);


    SimulationResult call_payoff_montecarlo(const unsigned int n_simulations);
    SimulationResult put_payoff_montecarlo(const unsigned int n_simulations);

    SimulationResult call_payoff_montecarlo_cpu(const unsigned int n_simulations);
    SimulationResult put_payoff_montecarlo_cpu(const unsigned int n_simulations);

    float call_payoff_blackSholes();
    float put_payoff_blackSholes();

    float stock_T();
};

#endif //OPTIONPRICING_EUROPEANOPTION_CUH
