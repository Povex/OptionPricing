//
// Created by marco on 23/05/22.
//

#include "EuropeanOption.cuh"

EuropeanOption::EuropeanOption(Asset *asset, float strikePrice, float timeToMaturity) : Option(asset),
                                                                                        strikePrice(strikePrice),
                                                                                        timeToMaturity(
                                                                                                  timeToMaturity) {}

float EuropeanOption::getStrikePrice() const {
    return strikePrice;
}

void EuropeanOption::setStrikePrice(float strikePrice) {
    EuropeanOption::strikePrice = strikePrice;
}

float EuropeanOption::getTimeToMaturity() const {
    return timeToMaturity;
}

void EuropeanOption::setTimeToMaturity(float timeToMaturity) {
    EuropeanOption::timeToMaturity = timeToMaturity;
}
