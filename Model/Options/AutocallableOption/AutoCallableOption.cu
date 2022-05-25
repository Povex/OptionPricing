//
// Created by marco on 24/05/22.
//

#include "AutoCallableOption.cuh"

AutoCallableOption::AutoCallableOption(Asset *asset, float rebase, const std::vector<float> &observationDates,
                                       const std::vector<float> &barriers, const std::vector<float> &payoffs) : Option(
        asset), rebase(rebase), observationDates(observationDates), barriers(barriers), payoffs(payoffs) {}

float AutoCallableOption::getRebase() const {
    return rebase;
}

void AutoCallableOption::setRebase(float rebase) {
    this->rebase = rebase;
}

const std::vector<float> &AutoCallableOption::getObservationDates() const {
    return observationDates;
}

void AutoCallableOption::setObservationDates(const std::vector<float> &observationDates) {
    this->observationDates = observationDates;
}

const std::vector<float> &AutoCallableOption::getBarriers() const {
    return barriers;
}

void AutoCallableOption::setBarriers(const std::vector<float> &barriers) {
    this->barriers = barriers;
}

const std::vector<float> &AutoCallableOption::getPayoffs() const {
    return payoffs;
}

void AutoCallableOption::setPayoffs(const std::vector<float> &payoffs) {
    this->payoffs = payoffs;
}
