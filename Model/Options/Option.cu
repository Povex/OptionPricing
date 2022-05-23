//
// Created by marco on 19/05/22.
//

#include "Option.cuh"


Option::Option(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams) : asset(asset),
                                                                                         gpuParams(gpuParams),
                                                                                         monteCarloParams(
                                                                                                 monteCarloParams) {}

Asset *Option::getAsset() const {
    return asset;
}

void Option::setAsset(Asset *asset) {
    Option::asset = asset;
}

GPUParams *Option::getGpuParams() const {
    return gpuParams;
}

void Option::setGpuParams(GPUParams *gpuParams) {
    Option::gpuParams = gpuParams;
}

MonteCarloParams *Option::getMonteCarloParams() const {
    return monteCarloParams;
}

void Option::setMonteCarloParams(MonteCarloParams *monteCarloParams) {
    Option::monteCarloParams = monteCarloParams;
}

Option::Option() {}

