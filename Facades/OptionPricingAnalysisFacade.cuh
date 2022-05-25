//
// Created by marco on 23/05/22.
//

#ifndef OPTIONPRICING_OPTIONPRICINGANALYSISFACADE_CUH
#define OPTIONPRICING_OPTIONPRICINGANALYSISFACADE_CUH

class OptionPricingAnalysisFacade {
public:
    void executeAnalysis();

    void europeanOptionsComparisonsImpl();

    void europeanOptionsErrorTrendSimulations();

    void europeanOptionTimeGpuParams();

    void europeanOptionsExecTimeMultipleOptions();

    void autoCallableAsymptoticLimitsAnalysis();
};


#endif //OPTIONPRICING_OPTIONPRICINGANALYSISFACADE_CUH
