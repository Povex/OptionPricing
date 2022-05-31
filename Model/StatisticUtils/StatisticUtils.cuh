//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_STATISTICUTILS_CUH
#define OPTIONPRICING_STATISTICUTILS_CUH


#include <vector>

class StatisticUtils {
protected:
    float stdError;
    float stdDev;
    std::vector<float> confidence;
    float mean;

public:
    StatisticUtils(float stdError, float stdDev, const std::vector<float> &confidence, float mean);

    StatisticUtils();

    ~StatisticUtils() = default;

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


#endif //OPTIONPRICING_STATISTICUTILS_CUH
