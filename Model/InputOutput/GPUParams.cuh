//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_GPUPARAMS_CUH
#define OPTIONPRICING_GPUPARAMS_CUH

class GPUParams {
protected:
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

public:
    GPUParams(const dim3 &threadsPerBlock, const dim3 &blocksPerGrid);

    ~GPUParams() = default;

    const dim3 &getThreadsPerBlock() const;

    void setThreadsPerBlock(const dim3 &threadsPerBlock);

    const dim3 &getBlocksPerGrid() const;

    void setBlocksPerGrid(const dim3 &blocksPerGrid);

};

#endif //OPTIONPRICING_GPUPARAMS_CUH
