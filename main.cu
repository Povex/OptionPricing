
#include "Model/InputOutput/Asset.cuh"
#include "Model/InputOutput/SimulationResult.cuh"
#include "Model/InputOutput/GPUParams.cuh"
#include "Model/InputOutput/MonteCarloParams.cuh"
#include "Facades/OptionPricingFacade.cuh"
#include "Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cuh"
#include "Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cuh"
#include "Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cuh"
#include "Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cuh"
#include "Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cuh"
#include "Facades/OptionPricingAnalysisFacade.cuh"
#include "Utils/ContextGPU.cuh"

#include <iostream>

using namespace std;

/*
void testEuropeanOption(){
    Asset asset(100.0f, 0.3f, 0.01f);
    EuropeanOption option(asset, 100.0f, 1.0f);

    unsigned int n_simulations = 10000000;

    cout << "Payoff call: " << option.call_payoff_blackSholes() << endl;
    cout << "Payoff put: " << option.put_payoff_blackSholes() << endl;

    clock_t start, end;
    start = clock();
    SimulationResult t = option.call_payoff_montecarlo(n_simulations);
    end = clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by GPU for call is : " << fixed
         << time_taken << std::setprecision(5);
    cout << " sec " << endl;
    cout << "Value " << t.getValue() << endl;
    cout << "Confidence [" << t.getConfidenceInterval()[0] << ", " << t.getConfidenceInterval()[1] << "]" << endl;
    cout << "Standard error " << t.getStdError() << endl;

    start = clock();
    t = option.call_payoff_montecarlo_cpu(n_simulations);
    end = clock();
    time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by CPU for call is : " << fixed
         << time_taken << std::setprecision(5);
    cout << " sec " << endl;
    cout << "Value " << t.getValue() << endl;
    cout << "Confidence [" << t.getConfidenceInterval()[0] << ", " << t.getConfidenceInterval()[1] << "]" << endl;
    cout << "Standard error " << t.getStdError() << endl;
}

void testBinaryOption(){
    Asset asset(100.0f, 0.3f, 0.02f);
    BinaryOptionToDelete option(asset, 115.0f, 0.25f, 5.0f);

    cout << "Actual call payoff motecarlo gpu: " << option.call_payoff_montecarlo_gpu(10000000) << endl;
    cout << "Actual put payoff montecarlo gpu: " << option.put_payoff_montecarlo_gpu(10000000) << endl;
    cout << "Actual call payoff blackSholes: " << option.actual_call_payoff_blackSholes() << endl;
    cout << "Actual put payoff blackSholes: " << option.actual_put_payoff_blackSholes() << endl;
}

void testAutoCallable(){
    Asset asset(100.0f, 0.3f, 0.3f);
    int n_binary_option = 3;

    std::vector<float> observationDates(n_binary_option);
    observationDates[0] = 0.2f;
    observationDates[1] = 0.4f;
    observationDates[2] = 1.0f;

    std::vector<float> barriers(n_binary_option);
    barriers[0] = 110.0f;
    barriers[1] = 140.0f;
    barriers[2] = 160.0f;

    std::vector<float> payoffs(n_binary_option);
    payoffs[0] = 10.0f;
    payoffs[1] = 20.0f;
    payoffs[2] = 40.0f;


    AutoCallableOption option(asset, 50.0f, observationDates, barriers, payoffs);

    clock_t start, end;
    start = clock();
    SimulationResult t = option.call_payoff_montecarlo_cpu();
    end = clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by CPU for call is : " << fixed
         << time_taken << std::setprecision(5);
    cout << " sec " << endl;
    cout << "Value " << t.getValue() << endl;
    cout << "Confidence [" << t.getConfidenceInterval()[0] << ", " << t.getConfidenceInterval()[1] << "]" << endl;
    cout << "Standard error " << t.getStdError() << endl;


    start = clock();
    t = option.call_payoff_montecarlo_gpu();
    end = clock();

    time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by GPU for call is : " << fixed
         << time_taken << std::setprecision(5);
    cout << " sec " << endl;
    cout << "Value " << t.getValue() << endl;
    cout << "Confidence [" << t.getConfidenceInterval()[0] << ", " << t.getConfidenceInterval()[1] << "]" << endl;
    cout << "Standard error " << t.getStdError() << endl;
}

 */

void europeanOptionTest(){
    int nSimulations = pow(2, 26);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams *monteCarloParams = new MonteCarloParams(nSimulations, CURAND_RNG_QUASI_SOBOL64, 42ULL);

    Asset *asset = new Asset(100.0f, 0.3f, 0.03f);
    EuropeanOptionGPU europeanOptiongpu(asset, 100.0f, 1.0f, monteCarloParams, &gpuParams);

    cout << "Call GPU simulation: " << europeanOptiongpu.callPayoff() << endl;

    EuropeanOptionAnalytical europeanOptionAnalytical(asset, 100.0, 1.0f);
    cout << "BlackSholes: " << (Result) europeanOptionAnalytical.callPayoff() << endl;

    EuropeanOptionSerialCPU europeanOptionSerialCpu(asset, 100.0f, 1.0f, monteCarloParams);
    cout << "Call CPU simulation: " << europeanOptionSerialCpu.callPayoff() << endl;

    /*s = europeanOption.callPayoffSerialCPU();
    cout << "Call CPU simulation: " << s << endl;

    cout << "Call Black-Sholes value: " << europeanOption.callPayoffBlackSholes() << endl << endl;


    s = europeanOption.putPayoff();
    cout << "Put GPU simulation: " << s << endl;
    s = europeanOption.putPayoffSerialCPU();
    cout << "Put CPU simulation: " << s << endl;

    cout << "Put Black-Sholes value: " << europeanOption.putPayoffBlackSholes() << endl;*/
}

/*
void testAutocallable2(){
    Asset asset(100.0f, 0.3f, 0.3f);
    int n_binary_option = 3;

    std::vector<float> observationDates(n_binary_option);
    observationDates[0] = 0.2f;
    observationDates[1] = 0.4f;
    observationDates[2] = 1.0f;

    std::vector<float> barriers(n_binary_option);
    barriers[0] = 110.0f;
    barriers[1] = 140.0f;
    barriers[2] = 160.0f;

    std::vector<float> payoffs(n_binary_option);
    payoffs[0] = 10.0f;
    payoffs[1] = 20.0f;
    payoffs[2] = 40.0f;

    GPUParams gpuParams(256);
    MonteCarloParams monteCarloParams(1e5, 0);

    AutocallableOption2 option(&asset, &gpuParams, &monteCarloParams, 50.0f, observationDates, barriers, payoffs);

    cout << "Call GPU simulation: " << option.callPayoff() << endl;
    cout << "Call CPU simulation: " << option.callPayoffMontecarloCpu() << endl;
}
*/

void testEuropeanOptions2(){
    OptionPricingFacade optionPricingFacade;
    vector<SimulationResult> results = optionPricingFacade.executeEuropeanCalls();

    for(const SimulationResult& result: results){
        cout << result << endl;
    }
}

void testAutoCallableOptions2(){
    Asset asset(100.0f, 0.3f, 0.3f);
    int n_binary_option = 365;

    std::vector<float> observationDates;
    std::vector<float> barriers;
    std::vector<float> payoffs;

    float dt = 1.0f/365;
    float currTime = dt;
    for (int i=0; i<n_binary_option; i++){
        observationDates.push_back(currTime);
        barriers.push_back(115.0f);
        payoffs.push_back(140.0f);

        currTime += dt;
    }

    int nSimulations = 1e6;
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_DEFAULT, 42ULL);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);

    AutoCallableOptionCPU optionC(&asset, 50.0f, observationDates, barriers, payoffs,&monteCarloParams);
    cout << "AutoCallable CPU: " << optionC.callPayoff() << endl;

    AutoCallableOptionGPU optionG(&asset, 50.0f, observationDates, barriers, payoffs,&monteCarloParams, &gpuParams);
    cout << "AutoCallable GPU: " << optionG.callPayoff() << endl;
}

int main() {
    ContextGPU context;
    context.printProperties();

    OptionPricingAnalysisFacade facade;
    facade.autoCallableCorrispondenceBinaryOption();

}
