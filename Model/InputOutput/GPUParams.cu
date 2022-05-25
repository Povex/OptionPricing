//
// Created by marco on 19/05/22.
//

#include "GPUParams.cuh"


GPUParams::GPUParams(const dim3 &threadsPerBlock, const dim3 &blocksPerGrid) : threadsPerBlock(threadsPerBlock),
                                                                               blocksPerGrid(blocksPerGrid) {}

const dim3 &GPUParams::getThreadsPerBlock() const {
    return threadsPerBlock;
}

void GPUParams::setThreadsPerBlock(const dim3 &threadsPerBlock) {
    this->threadsPerBlock = threadsPerBlock;
}

const dim3 &GPUParams::getBlocksPerGrid() const {
    return blocksPerGrid;
}

void GPUParams::setBlocksPerGrid(const dim3 &blocksPerGrid) {
    this->blocksPerGrid = blocksPerGrid;
}
