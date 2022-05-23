//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_STATISTICUTILSCPU_CUH
#define OPTIONPRICING_STATISTICUTILSCPU_CUH


#include "StatisticUtils.cuh"

class StatisticUtilsCPU : public StatisticUtils {
protected:
    float *samples;
    unsigned int n;

public:
    StatisticUtilsCPU(float *samples, unsigned int n);

    void calcMean();
    void calcCI();
};


#endif //OPTIONPRICING_STATISTICUTILSCPU_CUH
