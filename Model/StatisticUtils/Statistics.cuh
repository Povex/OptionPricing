//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_STATISTICS_CUH
#define OPTIONPRICING_STATISTICS_CUH


#include <vector>

class Statistics {
protected:
    float stdError;
    float stdDev;
    std::vector<float> confidence;
    float mean;

public:
    Statistics(float stdError, float stdDev, const std::vector<float> &confidence, float mean);

    Statistics();

    ~Statistics() = default;

    void calcMean();
    void calcCI();

    float getStdError() const;

    void setStdError(float stdError);

    float getStdDev() const;

    void setStdDev(float stdDev);

    const std::vector<float> &getConfidence() const;

    void setConfidence(const std::vector<float> &confidence);

    float getMean() const;

    void setMean(float mean);
};


#endif //OPTIONPRICING_STATISTICS_CUH
