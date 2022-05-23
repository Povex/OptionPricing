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

std::ostream &operator<<(std::ostream &os, const Result &result) {
    os << "value: " << result.value;
    return os;
}


