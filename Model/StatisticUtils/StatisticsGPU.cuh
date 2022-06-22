//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_STATISTICSGPU_CUH
#define OPTIONPRICING_STATISTICSGPU_CUH


#include <thrust/device_vector.h>
#include "Statistics.cuh"

class StatisticsGPU  : public Statistics {
protected:
    dim3 blockDim1D;
    dim3 gridDim1D;
    thrust::device_vector<float> samples;

public:
    StatisticsGPU(const dim3 &blockDim1D, const dim3 &gridDim1D, const thrust::device_vector<float> &samples);

    ~StatisticsGPU() = default;

    void calcMean();
    void calcCI();

    const dim3 &getBlockDim1D() const;

    void setBlockDim1D(const dim3 &blockDim1D);

    const dim3 &getGridDim1D() const;

    void setGridDim1D(const dim3 &gridDim1D);
};


#endif //OPTIONPRICING_STATISTICSGPU_CUH
