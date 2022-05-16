//
// Created by marco on 12/05/22.
//

#ifndef OPTIONPRICING_OPTIONPRICINGUTILS_H
#define OPTIONPRICING_OPTIONPRICINGUTILS_H

float next_price_exact(float currentPrice, float riskFreeRate, float volatility, float dt, float z){
    return currentPrice * exp((riskFreeRate - (pow(volatility, 2) / 2)) * dt + volatility * sqrt(dt) * z);
}

float next_price_eulero(float currentPrice, float riskFreeRate, float volatility, float dt, float z){
    return currentPrice * (1 + (riskFreeRate * dt) + (volatility * sqrt(dt) * z));
}

#endif //OPTIONPRICING_OPTIONPRICINGUTILS_H
