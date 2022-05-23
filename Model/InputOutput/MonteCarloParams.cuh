//
// Created by marco on 19/05/22.
//

#ifndef OPTIONPRICING_MONTECARLOPARAMS_CUH
#define OPTIONPRICING_MONTECARLOPARAMS_CUH


class MonteCarloParams {
private:
    int nSimulations;

    int PRNGType;
public:
    MonteCarloParams(int nSimulations, int prngType);

    ~MonteCarloParams() = default;

    int getNSimulations() const;

    void setNSimulations(int nSimulations);

    int getPrngType() const;

    void setPrngType(int prngType);

};


#endif //OPTIONPRICING_MONTECARLOPARAMS_CUH
