//
// Created by marco on 02/05/22.
//

#include "Asset.cuh"

Asset::Asset() {}

Asset::Asset(float spotPrice, float volatility, float riskFreeRate) : spot_price(spotPrice), volatility(volatility),
                                                                      risk_free_rate(riskFreeRate) {}

float Asset::getSpotPrice() const {
    return spot_price;
}

void Asset::setSpotPrice(float spotPrice) {
    spot_price = spotPrice;
}

float Asset::getVolatility() const {
    return volatility;
}

void Asset::setVolatility(float volatility) {
    this->volatility = volatility;
}

float Asset::getRiskFreeRate() const {
    return risk_free_rate;
}

void Asset::setRiskFreeRate(float riskFreeRate) {
    risk_free_rate = riskFreeRate;
}

Asset::~Asset() {

}
