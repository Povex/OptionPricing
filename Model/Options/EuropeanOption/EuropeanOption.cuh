//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTION_CUH
#define OPTIONPRICING_EUROPEANOPTION_CUH


#include "../Option.cuh"

class EuropeanOption : public Option{
protected:
    float strikePrice;

    float timeToMaturity;
public:
    EuropeanOption(Asset *asset, float strikePrice, float timeToMaturity);

    ~EuropeanOption() = default;

    float getStrikePrice() const;

    void setStrikePrice(float strikePrice);

    float getTimeToMaturity() const;

    void setTimeToMaturity(float timeToMaturity);
};


#endif //OPTIONPRICING_EUROPEANOPTION_CUH
