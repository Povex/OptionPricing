//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_MONTECARLOPARAMS_CUH
#define OPTIONPRICING_MONTECARLOPARAMS_CUH


#include <curand.h>

class MonteCarloParams {
protected:
    int nSimulations;

    curandRngType rngType;

    unsigned long long seed;

public:
    MonteCarloParams(int nSimulations, curandRngType rngType, unsigned long long int seed);

    ~MonteCarloParams() = default;

    int getNSimulations() const;

    void setNSimulations(int nSimulations);

    curandRngType getRngType() const;

    void setRngType(curandRngType rngType);

    unsigned long long int getSeed() const;

    void setSeed(unsigned long long int seed);

};


#endif //OPTIONPRICING_MONTECARLOPARAMS_CUH
