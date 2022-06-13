//
// Created by marco on 30/05/22.
//

#include "BinaryOptionAnalytical.cuh"
#include "../../Shared/SharedFunctions.cuh"


BinaryOptionAnalytical::BinaryOptionAnalytical(Asset *asset, float timeToMaturity, float barrier, float payoff,
                                               float rebase) : BinaryOption(asset, timeToMaturity, barrier, payoff,
                                                                            rebase) {}


SimulationResult BinaryOptionAnalytical::callPayoff() {
    double d1 = log(getAsset()->getSpotPrice() / barrier) + (getAsset()->getRiskFreeRate() + 0.5 * pow(getAsset()->getVolatility(), 2)) * timeToMaturity;
    d1 = d1 / (getAsset()->getVolatility() * sqrt(timeToMaturity));

    double d2 = d1 - getAsset()->getVolatility() * sqrt(timeToMaturity);

    double in_money_probability = normalCDF(d2) * exp(getAsset()->getRiskFreeRate() * timeToMaturity);
    double call_value = in_money_probability * exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    SimulationResult result(call_value);

    return result;
}

SimulationResult BinaryOptionAnalytical::putPayoff() {
    double d1 = log(getAsset()->getSpotPrice() / barrier) + (getAsset()->getRiskFreeRate() + 0.5 * pow(getAsset()->getVolatility(), 2)) * timeToMaturity;
    d1 = d1 / (getAsset()->getVolatility() * sqrtf(timeToMaturity));

    double d2 = d1 - getAsset()->getVolatility() * sqrt(timeToMaturity);

    double in_money_probability = normalCDF(-d2) * exp(getAsset()->getRiskFreeRate() * timeToMaturity);
    double putValue = in_money_probability * exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    SimulationResult result(putValue);

    return result;
}

