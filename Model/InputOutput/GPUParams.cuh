//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_GPUPARAMS_CUH
#define OPTIONPRICING_GPUPARAMS_CUH

class GPUParams {
private:
    int nThreads;

public:
    GPUParams(int nThreads);

    ~GPUParams() = default;

    int getNThreads();

    void setNThreads(int nThreads);
};

#endif //OPTIONPRICING_GPUPARAMS_CUH
