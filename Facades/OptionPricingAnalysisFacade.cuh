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

    void autoCallableNObservationDates();

    void autoCallableOptionsErrorTrendSimulation();

    void autoCallableExecTimeMultipleOptions();

    void autoCallablePriceDependenceBarriersConstant();

    void autoCallableErrorDependenceBarriersConstant();

    void autoCallablePriceDependenceBarriersLinearIncreasing();

    void autoCallablePriceDependenceBarriersLinearDecreasing();

    void autoCallableErrorDependenceNObsDates();

    void autoCallableCorrispondenceBinaryOption();

    void europeanOptionsErrorTrendSimulationsQM();
};


#endif //OPTIONPRICING_OPTIONPRICINGANALYSISFACADE_CUH
