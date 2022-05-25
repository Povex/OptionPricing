//
// Created by marco on 24/05/22.
//

#ifndef OPTIONPRICING_AUTOCALLABLEOPTION_CUH
#define OPTIONPRICING_AUTOCALLABLEOPTION_CUH


#include "../Option.cuh"

class AutoCallableOption : public Option{
protected:
    float rebase;
    std::vector<float> observationDates;
    std::vector<float> barriers;
    std::vector<float> payoffs;
public:
    AutoCallableOption(Asset *asset, float rebase, const std::vector<float> &observationDates,
                       const std::vector<float> &barriers, const std::vector<float> &payoffs);

    ~AutoCallableOption() = default;

    float getRebase() const;

    void setRebase(float rebase);

    const std::vector<float> &getObservationDates() const;

    void setObservationDates(const std::vector<float> &observationDates);

    const std::vector<float> &getBarriers() const;

    void setBarriers(const std::vector<float> &barriers);

    const std::vector<float> &getPayoffs() const;

    void setPayoffs(const std::vector<float> &payoffs);

};


#endif //OPTIONPRICING_AUTOCALLABLEOPTION_CUH
