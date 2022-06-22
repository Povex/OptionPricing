//
// Created by marco on 21/06/22.
//

#ifndef OPTIONPRICING_STATISTICSCPU_CUH
#define OPTIONPRICING_STATISTICSCPU_CUH


#include "Statistics.cuh"

class StatisticsCPU : public Statistics {
protected:
    float *samples;
    unsigned int n;

public:
    StatisticsCPU(float *samples, unsigned int n);

    ~StatisticsCPU() = default;

    void calcMean();
    void calcCI();
};


#endif //OPTIONPRICING_STATISTICSCPU_CUH
