//
// Created by marco on 23/05/22.
//

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

#include "OptionPricingAnalysisFacade.cuh"
#include "../Model/InputOutput/GPUParams.cuh"
#include "../Model/InputOutput/MonteCarloParams.cuh"
#include "../Model/InputOutput/Asset.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cuh"
#include "../Utils/DateUtils.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOption.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cuh"
#include "../Utils/ContextGPU.cuh"

using namespace std;

void OptionPricingAnalysisFacade::executeAnalysis() {
    europeanOptionsComparisonsImpl();
    europeanOptionsErrorTrendSimulations();
}

void OptionPricingAnalysisFacade::europeanOptionsComparisonsImpl() {
    cout << "\n\n [Running] - European option comparisons, please wait...\n";

    int nSimulations = 1e8;
    int threadsPerBlock = 512;
    int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
    GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

    MonteCarloParams monteCarloParams(nSimulations, 0);
    Asset asset(100.0f, 0.25f, 0.01f);

    float strikePrice = 100;
    float timeToMaturity = 1.0f;
    EuropeanOption *optionAnalytical, *optionSerialCPU, *optionGPU;

    optionAnalytical = new EuropeanOptionAnalytical(&asset, strikePrice, timeToMaturity);
    SimulationResult analyticalResultC = optionAnalytical->callPayoff();
    SimulationResult analyticalResultP = optionAnalytical->putPayoff();

    optionSerialCPU = new EuropeanOptionSerialCPU(&asset, strikePrice, timeToMaturity, &monteCarloParams);
    SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
    SimulationResult serialCpuResultP = optionSerialCPU->putPayoff();

    optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
    SimulationResult gpuResultC = optionGPU->callPayoff();
    SimulationResult gpuResultP = optionGPU->putPayoff();

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/EuropeanOption/ComparisonsImpl/Output/" + filename + ".csv");
    myFile << "Type,Engine,value,stdError,confidence1,confidence2,timeElapsed[s]\n";
    myFile << "EuropeanCall" << sep << "Analytical" << sep << analyticalResultC.getValue() << "\n";
    myFile << "EuropeanCall" << sep << "SerialCPU" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << "\n";
    myFile << "EuropeanCall" << sep << "GPU" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << "\n";
    myFile << "EuropeanPut" << sep << "Analytical" << sep << analyticalResultP.getValue() << "\n";
    myFile << "EuropeanPut" << sep << "SerialCPU" << sep << serialCpuResultP.getValue() << sep << serialCpuResultP.getStdError()
           << sep << serialCpuResultP.getConfidence()[0] << sep << serialCpuResultP.getConfidence()[1]
           << sep << serialCpuResultP.getTimeElapsed() << "\n";
    myFile << "EuropeanPut" << sep << "GPU" << sep << gpuResultP.getValue() << sep << gpuResultP.getStdError()
           << sep << gpuResultP.getConfidence()[0] << sep << gpuResultP.getConfidence()[1]
           << sep << gpuResultP.getTimeElapsed() << "\n";

    myFile.close();
}

void OptionPricingAnalysisFacade::europeanOptionsErrorTrendSimulations() {
    cout << "\n [Running] - European option error trend simulations, please wait...\n";

    dim3 threadsPerBlock(512);
    Asset asset(100.0f, 0.25f, 0.01f);
    float strikePrice = 100.0f;
    float timeToMaturity = 1.0f;

    EuropeanOption *optionAnalytical, *optionSerialCPU, *optionGPU;
    MonteCarloParams *monteCarloParams;

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/EuropeanOption/ErrorTrendSimulations/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,value,stdError,confidence1,confidence2,timeElapsed[s]\n";

    optionAnalytical = new EuropeanOptionAnalytical(&asset, strikePrice, timeToMaturity);
    SimulationResult analyticalC = optionAnalytical->callPayoff();
    SimulationResult analyticalP = optionAnalytical->putPayoff();

    myFile << "EuropeanCall" << sep << "Analytical" << sep << sep << analyticalC.getValue() << "\n";
    myFile << "EuropeanPut" << sep << "Analytical" << sep << sep << analyticalP.getValue() << "\n";

    int nSimulations = 0;
    for(int i=0; i<28; i++){
        nSimulations = pow(2, i);
        monteCarloParams = new MonteCarloParams(nSimulations, 0);
        dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);

        GPUParams gpuParams(threadsPerBlock, blocksPerGrid);

        optionSerialCPU = new EuropeanOptionSerialCPU(&asset, strikePrice, timeToMaturity, monteCarloParams);
        SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
        SimulationResult serialCpuResultP = optionSerialCPU->putPayoff();

        optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();
        SimulationResult gpuResultP = optionGPU->putPayoff();

        string gpuName = ContextGPU::instance()->getDeviceProp().name;
        string date = DateUtils().getDate();
        string filename = gpuName + " " + date;
        string sep = ",";

        myFile << "EuropeanCall" << sep << "SerialCPU" << sep << nSimulations << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
               << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
               << sep << serialCpuResultC.getTimeElapsed() << "\n";
        myFile << "EuropeanCall" << sep << "GPU" << sep << nSimulations << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";
        myFile << "EuropeanPut" << sep << "SerialCPU" << sep << nSimulations << sep << serialCpuResultP.getValue() << sep << serialCpuResultP.getStdError()
               << sep << serialCpuResultP.getConfidence()[0] << sep << serialCpuResultP.getConfidence()[1]
               << sep << serialCpuResultP.getTimeElapsed() << "\n";
        myFile << "EuropeanPut" << sep << "GPU" << sep << nSimulations << sep << gpuResultP.getValue() << sep << gpuResultP.getStdError()
               << sep << gpuResultP.getConfidence()[0] << sep << gpuResultP.getConfidence()[1]
               << sep << gpuResultP.getTimeElapsed() << "\n";
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::europeanOptionsExecTimeMultipleOptions(){
    cout << "\n [Running] - European option batch simulations, please wait...\n";

    int nSimulations = pow(2, 24);;
    int threadsPerBlock = 512;
    int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
    GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

    MonteCarloParams monteCarloParams(nSimulations, 0);
    Asset asset(100.0f, 0.25f, 0.01f);
    float strikePrice = 100.0f;
    float timeToMaturity = 1.0f;

    EuropeanOption *optionSerialCPU, *optionGPU;
    optionSerialCPU = new EuropeanOptionSerialCPU(&asset, strikePrice, timeToMaturity, &monteCarloParams);
    optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/EuropeanOption/ExecutionTimeMultipleOptions/Output/" + filename + ".csv");
    myFile << "Type,Engine,batchSize,nSimulationsPerOption,timeElapsed[s]\n";

    int nBatch = 0;
    double batchTimeCPU = 0.0;
    double batchTimeGPU = 0.0;

    SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
    double singleTimeCPU = serialCpuResultC.getTimeElapsed();
    SimulationResult gpuResultC = optionGPU->callPayoff();
    double singleTimeGPU = gpuResultC.getTimeElapsed();

    // Simulate the batch size 2^0, 2^1, .., 2^10
    for(int i=0; i<=11; i++) {
        nBatch = pow(2, i);
        batchTimeCPU = 0.0;
        batchTimeGPU = 0.0;

        batchTimeCPU = singleTimeCPU * nBatch;
        batchTimeGPU = singleTimeGPU * nBatch;

        myFile << "EuropeanCall" << sep << "SerialCPU" << sep << nBatch << sep << nSimulations << sep << batchTimeCPU << "\n";
        myFile << "EuropeanCall" << sep << "GPU" << sep << nBatch << sep << nSimulations << sep << batchTimeGPU << "\n";
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::europeanOptionTimeGpuParams() {
    cout << "\n [Running] - European option time for different gpu parameters (GridDim, BlockDim), please wait...\n";

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/EuropeanOption/ExecutionTimeGPUParams/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,blockDim,timeElapsed[s]\n";

    for(int i=1; i<=32; i++){
        int nSimulations = pow(2, 24);
        dim3 threadsPerBlock(32 * i);
        dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);

        GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
        MonteCarloParams monteCarloParams(nSimulations, 0);

        cout << "blockDim" << threadsPerBlock.x << " GridDim" << blocksPerGrid.x << endl;

        Asset asset(100.0f, 0.25f, 0.01f);
        float strikePrice = 100.0f;
        float timeToMaturity = 1.0f;

        EuropeanOption *optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity,
                                                          &monteCarloParams, &gpuParams);
        SimulationResult gpuResult = optionGPU->callPayoff();
        myFile << "EuropeanCall" << sep << "GPU" << sep << nSimulations << sep << gpuParams.getThreadsPerBlock().x << sep << gpuResult.getTimeElapsed() << "\n";
    }

    myFile.close();
}


void OptionPricingAnalysisFacade::autoCallableAsymptoticLimitsAnalysis() {
    cout << "\n [Running] - AutoCallable option asymptotic limits analysis, please wait...\n";

    int nSimulations = pow(2, 24);
    int threadsPerBlock = 512;
    int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
    GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

    // barriers[0] = 0 -> payoff = discounted(payoffs[0])
    MonteCarloParams monteCarloParams(nSimulations, 0);
    Asset asset(100.0f, 0.3f, 0.03f);

    int n_binary_option = 3;
    std::vector<float> observationDates(n_binary_option);
    observationDates[0] = 0.2f;
    observationDates[1] = 0.4f;
    observationDates[2] = 1.0f;

    std::vector<float> barriers(n_binary_option);
    barriers[0] = 0.0f;
    barriers[1] = 140.0f;
    barriers[2] = 160.0f;

    std::vector<float> payoffs(n_binary_option);
    payoffs[0] = 10.0f;
    payoffs[1] = 20.0f;
    payoffs[2] = 40.0f;

    double expected = payoffs[0] * exp(-0.03f * observationDates[0]);

    AutoCallableOption *optionSerialCPU, *optionGPU;
    optionSerialCPU = new AutoCallableOptionCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
    SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();

    optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
    SimulationResult gpuResultC = optionGPU->callPayoff();
    string gpuName = ContextGPU::instance()->getDeviceProp().name;

    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/AutoCallableOption/AsymptoticLimitsAnalysis/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,AsyntoticType,value,stdError, confidence1, confidence2,timeElapsed, expected\n";
    myFile << "AutoCallableCall" << sep << nSimulations << sep <<"SerialCPU" << sep << "FirstBarrierIsZero" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << sep << expected << "\n";
    myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << "FirstBarrierIsZero" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << sep << expected << "\n";

    barriers[0] = std::numeric_limits<float>::infinity();
    barriers[1] = 0.0f;
    barriers[2] = 160.0f;

    expected = payoffs[1] * exp(-0.03f * observationDates[1]);

    optionSerialCPU = new AutoCallableOptionCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
    serialCpuResultC = optionSerialCPU->callPayoff();

    optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
    gpuResultC = optionGPU->callPayoff();

    myFile << "AutoCallableCall" << sep << nSimulations << sep <<"SerialCPU" << sep << "SecondBarrierIsZero" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << sep << expected << "\n";
    myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << "SecondBarrierIsZero" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << sep << expected << "\n";

    barriers[0] = std::numeric_limits<float>::infinity();
    barriers[1] = std::numeric_limits<float>::infinity();
    barriers[2] = 0.0f;

    optionSerialCPU = new AutoCallableOptionCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
    serialCpuResultC = optionSerialCPU->callPayoff();

    optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
    gpuResultC = optionGPU->callPayoff();

    expected = payoffs[2] * exp(-0.03f * observationDates[2]);

    myFile << "AutoCallableCall" << sep << nSimulations << sep <<"SerialCPU" << sep << "LastBarrierIsZero" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << sep << expected << "\n";
    myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << "LastBarrierIsZero" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << sep << expected << "\n";


    barriers[0] = std::numeric_limits<float>::infinity();
    barriers[1] = std::numeric_limits<float>::infinity();
    barriers[2] = std::numeric_limits<float>::infinity();

    optionSerialCPU = new AutoCallableOptionCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
    serialCpuResultC = optionSerialCPU->callPayoff();

    optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
    gpuResultC = optionGPU->callPayoff();

    expected = 50.0f * exp(-0.03f * observationDates[2]);

    myFile << "AutoCallableCall" << sep << nSimulations << sep <<"SerialCPU" << sep << "AllBarriersInf" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << sep << expected << "\n";
    myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << "AllBarriersInf" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << sep << expected << "\n";

    myFile.close();
}



