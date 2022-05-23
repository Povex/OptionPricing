//
// Created by marco on 22/05/22.
//

#include "Result.cuh"

float Result::getValue() const {
    return value;
}

void Result::setValue(float value) {
    Result::value = value;
}

Result::Result(float value) : value(value) {}

Result::Result() {}


