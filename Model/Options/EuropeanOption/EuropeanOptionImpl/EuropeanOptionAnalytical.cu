//
// Created by marco on 23/05/22.
//

#include "EuropeanOptionAnalytical.cuh"

EuropeanOptionAnalytical::EuropeanOptionAnalytical(Asset *asset, float strikePrice, float timeToMaturity)
        : EuropeanOption(asset, strikePrice, timeToMaturity) {}

SimulationResult EuropeanOptionAnalytical::callPayoff() {
    float b = exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    float x1 = log(getAsset()->getSpotPrice() / (b * strikePrice)) + 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x1 = x1/(getAsset()->getVolatility()* (pow(timeToMaturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = z1 * getAsset()->getSpotPrice();

    float x2 = log(getAsset()->getSpotPrice() / (b * strikePrice)) - 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x2 = x2/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = b * strikePrice * z2;

    SimulationResult result(z1 - z2);

    return result;
}

SimulationResult EuropeanOptionAnalytical::putPayoff() {
    float b = exp(-getAsset()->getRiskFreeRate() * timeToMaturity);

    float x1 = log( (b * strikePrice)/getAsset()->getSpotPrice()) + 0.5 * (pow(getAsset()->getVolatility(), 2) * getTimeToMaturity());
    x1 = x1/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z1 = 0.5 * erfc(-x1 * M_SQRT1_2);
    z1 = b * getAsset()->getSpotPrice() * z1;

    float x2 = log((b * strikePrice)/getAsset()->getSpotPrice()) - 0.5 * (pow(getAsset()->getVolatility(), 2) * timeToMaturity);
    x2 = x2/(getAsset()->getVolatility() * (pow(timeToMaturity, 0.5)));

    float z2 = 0.5 * erfc(-x2 * M_SQRT1_2);
    z2 = strikePrice * z2;

    SimulationResult result(z1 - z2);

    return result;
}
