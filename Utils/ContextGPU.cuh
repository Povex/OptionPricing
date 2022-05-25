//
// Created by marco on 24/05/22.
//

#include <string>
#include <ostream>

#ifndef OPTIONPRICING_CONTEXTGPU_CUH
#define OPTIONPRICING_CONTEXTGPU_CUH

using namespace std;

class ContextGPU {
protected:
    static ContextGPU* pInstance;
    int nDevices;
    int currentDevice;
    int driverVersion;
    int runtimeVersion;
    cudaDeviceProp deviceProp;


public:
    static ContextGPU* instance();

    ContextGPU();

    const cudaDeviceProp &getDeviceProp() const;

    void setDeviceProp(const cudaDeviceProp &deviceProp);

    void printProperties();

    dim3 getOptimalBlocksPerGrid(dim3 threadsPerBlock, unsigned int nSimulations);
};

#endif //OPTIONPRICING_CONTEXTGPU_CUH
