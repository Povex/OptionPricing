//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_STATISTICSSERIALCPU_CUH
#define OPTIONPRICING_STATISTICSSERIALCPU_CUH


#include "Statistics.cuh"

class StatisticsSerialCPU : public Statistics {
protected:
    float *samples;
    unsigned int n;

public:
    StatisticsSerialCPU(float *samples, unsigned int n);

    void calcMean();
    void calcCI();
};


#endif //OPTIONPRICING_STATISTICSSERIALCPU_CUH
