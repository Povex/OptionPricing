//
// Created by marco on 22/05/22.
//

#ifndef OPTIONPRICING_RESULT_CUH
#define OPTIONPRICING_RESULT_CUH


class Result {
private:
    float value;
public:
    Result(float value);

    Result();

    ~Result() = default;

    float getValue() const;

    void setValue(float value);
};

#endif //OPTIONPRICING_RESULT_CUH
