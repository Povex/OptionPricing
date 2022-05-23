//
// Created by marco on 05/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTION_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTION_CUH

#include <thrust/host_vector.h>

#include "../../Context/Asset.cuh"
#include "../../Context/SimulationResult.cuh"

#include <map>
#include <vector>

using namespace std;

class AutoCallableOption {

public:
    Asset asset;
    float rebase; // Redemption payment depending on final underlying price
    std::vector<float> observationDates;
    std::vector<float> barriers;
    std::vector<float> payoffs;
    map<float, float> dateBarrier;

    int n_intervals;
    float dt;

    AutoCallableOption();
    AutoCallableOption(Asset market, float rebase, vector<float>, vector<float>, vector<float>);
    AutoCallableOption(AutoCallableOption &option);

    SimulationResult call_payoff_montecarlo_gpu();
    SimulationResult call_payoff_montecarlo_cpu();

    void addDateBarrier(pair<float, float> p);
    void print_map();
};

#endif //OPTIONPRICING_AUTOCALLABLEOPTION_CUH
