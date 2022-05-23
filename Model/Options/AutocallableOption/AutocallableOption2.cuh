//
// Created by marco on 22/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTION2_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTION2_CUH


#include "../Option.cuh"

class AutocallableOption2 : public Option {

private:
    float rebase;
    std::vector<float> observationDates;
    std::vector<float> barriers;
    std::vector<float> payoffs;

public:
    AutocallableOption2(Asset *asset, GPUParams *gpuParams, MonteCarloParams *monteCarloParams, float rebase,
                        const std::vector<float> observationDates, const std::vector<float> barriers,
                        const std::vector<float> payoffs);

    SimulationResult callPayoff();
    SimulationResult putPayoff();

    SimulationResult callPayoffMontecarloCpu();

    float getRebase() const;

    void setRebase(float rebase);

    const std::vector<float> &getObservationDates() const;

    void setObservationDates(const std::vector<float> &observationDates);

    const std::vector<float> &getBarriers() const;

    void setBarriers(const std::vector<float> &barriers);

    const std::vector<float> &getPayoffs() const;

    void setPayoffs(const std::vector<float> &payoffs);

};


#endif //OPTIONPRICING_AUTOCALLABLEOPTION2_CUH
