//
// Created by marco on 19/05/22.
//

#include "GPUParams.cuh"

void GPUParams::setNThreads(int nThreads) {
    this->nThreads = nThreads;
}

int GPUParams::getNThreads() {
    return nThreads;
}

GPUParams::GPUParams(int nThreads) : nThreads(nThreads) {}




