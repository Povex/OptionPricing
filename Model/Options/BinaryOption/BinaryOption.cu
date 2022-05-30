//
// Created by marco on 30/05/22.
//

#include "BinaryOption.cuh"


float BinaryOption::getTimeToMaturity() const {
    return timeToMaturity;
}

void BinaryOption::setTimeToMaturity(float timeToMaturity) {
    this->timeToMaturity = timeToMaturity;
}

float BinaryOption::getPayoff() const {
    return payoff;
}

void BinaryOption::setPayoff(float payoff) {
    this->payoff = payoff;
}

float BinaryOption::getRebase() const {
    return rebase;
}

void BinaryOption::setRebase(float rebase) {
    this->rebase = rebase;
}

BinaryOption::BinaryOption(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase) : Option(
        asset), timeToMaturity(timeToMaturity), barrier(barrier), payoff(payoff), rebase(rebase) {}
