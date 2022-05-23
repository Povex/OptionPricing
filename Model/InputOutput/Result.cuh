//
// Created by marco on 22/05/22.
//

#ifndef OPTIONPRICING_RESULT_CUH
#define OPTIONPRICING_RESULT_CUH


#include <ostream>

class Result {
private:
    float value;
public:
    Result(float value);

    Result();

    ~Result() = default;

    float getValue() const;

    void setValue(float value);

    friend std::ostream &operator<<(std::ostream &os, const Result &result);
};

#endif //OPTIONPRICING_RESULT_CUH
