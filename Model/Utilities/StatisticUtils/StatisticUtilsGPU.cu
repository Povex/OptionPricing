//
// Created by marco on 23/05/22.
//

#include <thrust/reduce.h>
#include "StatisticUtilsGPU.cuh"
#include "../errorHandler.cu"

/*
//define transformation f(x) -> x^2
 struct square {
 __host__ __device__ floatoperator()(float x)
 { return x * x;     }
 };
 float snrm2_fast(device_vector<float>& x)
 { // with fusion return sqrt
 (
 transform_reduce(x.begin(), x.end(), square(), 0.0f, plus<float>()); }
*/

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
    double sum = thrust::reduce(samples.begin(), samples.end(), 0.0, thrust::plus<double>());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double mean = sum/samples.size();

    setMean(mean);

}

void StatisticUtilsGPU::calcCI() {
    unsigned int n = samples.size();
    float *ptr_samples = thrust::raw_pointer_cast(samples.data());
    // Use reduction to calculate variance
    varianceKernel<<<gridDim1D, blockDim1D>>>(ptr_samples, n, mean);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double variance_sum = thrust::reduce(samples.begin(), samples.end(), 0.0, thrust::plus<double>());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Calculate standard error
    double stdDev = sqrt(variance_sum/(n-1));
    double stdError = stdDev/(sqrt(n));
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
