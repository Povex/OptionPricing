//
// Created by marco on 23/05/22.
//

#include <thrust/reduce.h>
#include "StatisticUtilsGPU.cuh"
#include "../errorHandler.cu"


__global__
void varianceKernel(float *samples, int n, float mean){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while(i < n) {
        samples[i] = powf(samples[i] - mean, 2);
        i += blockDim.x * gridDim.x;
    }
}


void StatisticUtilsGPU::calcMean() {
    // Use reduction to calculate mean
    float sum = thrust::reduce(samples.begin(), samples.end(), 0.0f, thrust::plus<float>());
    float mean = sum/(float) samples.size();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    setMean(mean);

}

void StatisticUtilsGPU::calcCI() {
    unsigned int n = samples.size();
    float *ptr_samples = thrust::raw_pointer_cast(samples.data());
    // Use reduction to calculate variance
    varianceKernel<<<gridDim1D, blockDim1D>>>(ptr_samples, n, mean);
    float variance_sum = thrust::reduce(samples.begin(), samples.end(), 0.0f, thrust::plus<float>());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Calculate standard error
    float stdDev = sqrtf(variance_sum/float(n-1));
    float stdError = stdDev/(sqrtf(n));
    setStdError(stdError);
    setStdDev(stdDev);

    // Calculate confidence interval 95%
    std::vector<float> confidence(2);
    confidence[0] = mean - 1.96 * stdError;
    confidence[1] = mean + 1.96 * stdError;
    setConfidence(confidence);
}

const dim3 &StatisticUtilsGPU::getBlockDim1D() const {
    return blockDim1D;
}

void StatisticUtilsGPU::setBlockDim1D(const dim3 &blockDim1D) {
    this->blockDim1D = blockDim1D;
}

const dim3 &StatisticUtilsGPU::getGridDim1D() const {
    return gridDim1D;
}

void StatisticUtilsGPU::setGridDim1D(const dim3 &gridDim1D) {
    this->gridDim1D = gridDim1D;
}

StatisticUtilsGPU::StatisticUtilsGPU(const dim3 &blockDim1D, const dim3 &gridDim1D,
                                     const thrust::device_vector<float> &samples) : blockDim1D(blockDim1D),
                                                                                    gridDim1D(gridDim1D),
                                                                                    samples(samples) {}
