//
// Created by marco on 19/05/22.
//

#include "Option.cuh"

Option::Option() {}

Option::Option(Asset *asset) : asset(asset) {}

Asset *Option::getAsset() const {
    return asset;
}

void Option::setAsset(Asset *asset) {
    Option::asset = asset;
}



