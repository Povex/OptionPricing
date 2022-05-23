//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_EUROPEANOPTION_CUH
#define OPTIONPRICING_EUROPEANOPTION_CUH

#include "../Option.cuh"

class EuropeanOption : public Option {
private:
    float strikePrice;

    float timeToMaturity;
public:
    EuropeanOption(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams);

    EuropeanOption(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams, float strikePrice,
                   float timeToMaturity);

    ~EuropeanOption() = default;

    float getStrikePrice() const;

    void setStrikePrice(float strikePrice);

    float getTimeToMaturity() const;

    void setTimeToMaturity(float timeToMaturity);

    SimulationResult callPayoff() override;

    SimulationResult putPayoff() override;

    SimulationResult callPayoffSerialCPU();

    SimulationResult putPayoffSerialCPU();

    float callPayoffBlackSholes();

    float putPayoffBlackSholes();

};


#endif //OPTIONPRICING_EUROPEANOPTION_CUH
