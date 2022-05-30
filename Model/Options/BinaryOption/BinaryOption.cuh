//
// Created by marco on 30/05/22.
//

#ifndef OPTIONPRICING_BINARYOPTION_CUH
#define OPTIONPRICING_BINARYOPTION_CUH


#include "../Option.cuh"

class BinaryOption : public Option {
protected:
    float timeToMaturity; // [years]
    float barrier;
    float payoff;
    float rebase;
public:
    BinaryOption(Asset *asset, float timeToMaturity, float barrier, float payoff, float rebase);

    ~BinaryOption() = default;

    float getTimeToMaturity() const;

    void setTimeToMaturity(float timeToMaturity);

    float getPayoff() const;

    void setPayoff(float payoff);

    float getRebase() const;

    void setRebase(float rebase);

};


#endif //OPTIONPRICING_BINARYOPTION_CUH
