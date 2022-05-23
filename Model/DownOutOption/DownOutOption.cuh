//
// Created by marco on 03/05/22.
//

#ifndef OPTIONPRICING_DOWNOUTOPTION_CUH
#define OPTIONPRICING_DOWNOUTOPTION_CUH

#include "../../Context/Asset.cuh"
#include "../../Context/SimulationResult.cuh"

class DownOutOption {

private:
    Asset asset;
    float strike_price;
    float time_to_maturity;
    int n_intervals;
    float dt;
    float barrier;
    float rebase;

public:
    DownOutOption();
    DownOutOption(Asset market, float strike_price, float time_to_maturity, float barrier);

    SimulationResult call_payoff();
    SimulationResult put_payoff();

    SimulationResult call_payoff_cpu();
};

#endif //OPTIONPRICING_DOWNOUTOPTION_CUH
