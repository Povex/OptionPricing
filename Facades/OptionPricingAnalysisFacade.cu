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
#include "../Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionSerialCPU.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cuh"
#include "../Utils/ContextGPU.cuh"
#include "../Model/Options/BinaryOption/BinaryOption.cuh"
#include "../Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionAnalytical.cuh"
#include "../Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionGPU.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionCPU.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cuh"

using namespace std;

void OptionPricingAnalysisFacade::executeAnalysis() {
    europeanOptionsComparisonsImpl();
    europeanOptionsErrorTrendSimulations();
    europeanOptionsExecTimeMultipleOptions();
    europeanOptionTimeGpuParams();

    autoCallableAsymptoticLimitsAnalysis();
}

void OptionPricingAnalysisFacade::europeanOptionsComparisonsImpl() {
    cout << "\n\n [Running] - European option comparisons, please wait...\n";

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/EuropeanOption/ComparisonsImpl/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,value,stdError,confidence1,confidence2,timeElapsed[s]\n";

    EuropeanOption *optionSerialCPU, *optionGPU, *optionCPU;

    float strikePrice = 100;
    float timeToMaturity = 1.0f;
    Asset asset(100.0f, 0.25f, 0.01f);

    int nSimulations = 0;

    for (int i=0; i < 9; i++){
        nSimulations = pow(10, i);
        dim3 threadsPerBlock(512);
        dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
        GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
        MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

        optionSerialCPU = new EuropeanOptionSerialCPU(&asset, strikePrice, timeToMaturity, &monteCarloParams);
        SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
        SimulationResult serialCpuResultP = optionSerialCPU->putPayoff();

        optionCPU = new EuropeanOptionCPU(&asset, strikePrice, timeToMaturity, &monteCarloParams);
        SimulationResult cpuResultC = optionCPU->callPayoff();
        SimulationResult cpuResultP = optionCPU->putPayoff();

        optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();
        SimulationResult gpuResultP = optionGPU->putPayoff();


        myFile << "EuropeanCall" << sep << "SerialCPU" << sep << nSimulations << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
               << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
               << sep << serialCpuResultC.getTimeElapsed() << "\n";
        myFile << "EuropeanCall" << sep << "GPU" << sep << nSimulations << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";
        myFile << "EuropeanCall" << sep << "CPU" << sep << nSimulations << sep << cpuResultC.getValue() << sep << cpuResultC.getStdError()
               << sep << cpuResultC.getConfidence()[0] << sep << cpuResultC.getConfidence()[1]
               << sep << cpuResultC.getTimeElapsed() << "\n";
        myFile << "EuropeanPut" << sep << "SerialCPU" << sep << nSimulations << sep << serialCpuResultP.getValue() << sep << serialCpuResultP.getStdError()
               << sep << serialCpuResultP.getConfidence()[0] << sep << serialCpuResultP.getConfidence()[1]
               << sep << serialCpuResultP.getTimeElapsed() << "\n";
        myFile << "EuropeanPut" << sep << "GPU" << sep << nSimulations << sep << gpuResultP.getValue() << sep << gpuResultP.getStdError()
               << sep << gpuResultP.getConfidence()[0] << sep << gpuResultP.getConfidence()[1]
               << sep << gpuResultP.getTimeElapsed() << "\n";
        myFile << "EuropeanPut" << sep << "CPU" << sep << nSimulations << sep << cpuResultP.getValue() << sep << cpuResultP.getStdError()
               << sep << cpuResultP.getConfidence()[0] << sep << cpuResultP.getConfidence()[1]
               << sep << cpuResultP.getTimeElapsed() << "\n";
    }

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
    for(int i=0; i<26; i++){
        nSimulations = pow(2, i);
        cout << "2 ^ " << i << endl;
        monteCarloParams = new MonteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);
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

    int nSimulations = pow(2, 26);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);

    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);
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
        MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

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

void OptionPricingAnalysisFacade::europeanOptionsErrorTrendSimulationsQM() {
    cout << "\n [Running] - European option error trend simulations quasi montecarlo, please wait...\n";

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
    std::ofstream myFile("AnalysisData/EuropeanOption/ErrorTrendSimulationsQM/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,value,stdError,confidence1,confidence2,timeElapsed[s]\n";

    optionAnalytical = new EuropeanOptionAnalytical(&asset, strikePrice, timeToMaturity);
    SimulationResult analyticalC = optionAnalytical->callPayoff();
    SimulationResult analyticalP = optionAnalytical->putPayoff();

    myFile << "EuropeanCall" << sep << "Analytical" << sep << sep << analyticalC.getValue() << "\n";
    myFile << "EuropeanPut" << sep << "Analytical" << sep << sep << analyticalP.getValue() << "\n";

    int nSimulations = 0;
    for(int i=0; i<26; i++){
        nSimulations = pow(2, i);
        cout << "2 ^ " << i << endl;
        monteCarloParams = new MonteCarloParams(nSimulations, CURAND_RNG_QUASI_SOBOL64, 42ULL);
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



void OptionPricingAnalysisFacade::autoCallableAsymptoticLimitsAnalysis() {
    cout << "\n [Running] - AutoCallable option asymptotic limits analysis, please wait...\n";

    int nSimulations = pow(2, 24);
    int threadsPerBlock = 512;
    int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
    GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

    // barriers[0] = 0 -> payoff = discounted(payoffs[0])
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);
    Asset asset(100.0f, 0.3f, 0.03f);

    int n_binary_option = 3;
    std::vector<float> observationDates(n_binary_option);
    observationDates[0] = 0.2f;
    observationDates[1] = 0.4f;
    observationDates[2] = 1.0f;

    std::vector<float> barriers(n_binary_option);
    barriers[0] = 0.0f;
    barriers[1] = 1.0f;
    barriers[2] = 1.0f;

    std::vector<float> payoffs(n_binary_option);
    payoffs[0] = 10.0f;
    payoffs[1] = 20.0f;
    payoffs[2] = 30.0f;

    double expected = payoffs[0] * exp(-0.03f * observationDates[0]);

    AutoCallableOption *optionSerialCPU, *optionGPU;
    optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
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
    barriers[2] = 1.0f;

    expected = payoffs[1] * exp(-0.03f * observationDates[1]);

    optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
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

    optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
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

    optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
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

void OptionPricingAnalysisFacade::autoCallableNObservationDates() {
    cout << "\n [Running] - AutoCallable option time varying the number of observation dates, please wait...\n";

    int nSimulations = 1e6;
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);

    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);
    Asset asset(100.0f, 0.3f, 0.03f);

    AutoCallableOption *optionSerialCPU, *optionGPU;

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/AutoCallableOption/TimeVaryingObservationDates/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,nBinaryOptions,value,stdError, confidence1, confidence2,timeElapsed, expected\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    float dailyDt = 1.0f/365;
    float currTime = 0.0f;
    float expected;
    double maxCpuTime = -1.0;

    for (int i = 1; i < 365; i++){ // n observation dates i
        cout << "N obs date: " << i << endl;
        int nBinaryOptions = i;
        observationDates.push_back(dailyDt);
        payoffs.push_back(10.0f);
        barriers.push_back(std::numeric_limits<float>::infinity());

        expected = 50.0f * exp(-0.03f * observationDates.back());

        if(nBinaryOptions < 128) {
            optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs,
                                                        &monteCarloParams);
            SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
            if(maxCpuTime < serialCpuResultC.getTimeElapsed()) maxCpuTime = serialCpuResultC.getTimeElapsed();

            myFile << "AutoCallableCall" << sep << nSimulations << sep << "SerialCPU" << sep << nBinaryOptions
                       << sep
                       << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
                       << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
                       << sep << serialCpuResultC.getTimeElapsed() << sep << expected << "\n";
        }

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();

        myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << nBinaryOptions << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << sep << expected << "\n";

        currTime += dailyDt;
    }
}

void OptionPricingAnalysisFacade::autoCallableOptionsErrorTrendSimulation() {
    cout << "\n [Running] - AutoCallable option error trend simulation, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    AutoCallableOption *optionSerialCPU, *optionCPU, *optionGPU;

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/AutoCallableOption/ErrorTrendSimulation/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,value,stdError,confidence1,confidence2,timeElapsed[s]\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float currTime = monthDt;

    for (int i = 1; i <= nBinaryOptions; i++){
        observationDates.push_back(currTime);
        payoffs.push_back(10.0f);
        barriers.push_back(1.2f);

        currTime += monthDt;
    }

    int nSimulations = 0;
    for (int i=0; i<25; i++){
        nSimulations = pow(2, i);
        dim3 threadsPerBlock(512);
        dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
        GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
        MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

        optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs,
                                                    &monteCarloParams);
        SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();

        optionCPU = new AutoCallableOptionCPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams);
        SimulationResult cpuResultC = optionCPU->callPayoff();

        myFile << "AutoCallableCall" << sep << "SerialCPU" << sep << "2^" + to_string(i) << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
               << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
               << sep << serialCpuResultC.getTimeElapsed() << "\n";
        myFile << "AutoCallableCall" << sep << "CPU" << sep << "2^" + to_string(i) << sep << cpuResultC.getValue() << sep << cpuResultC.getStdError()
               << sep << cpuResultC.getConfidence()[0] << sep << cpuResultC.getConfidence()[1]
               << sep << cpuResultC.getTimeElapsed() << "\n";
        myFile << "AutoCallableCall" << sep << "GPU" << sep << "2^" + to_string(i) << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";
    }

    myFile.close();

}

void OptionPricingAnalysisFacade::autoCallableExecTimeMultipleOptions() {
    cout << "\n [Running] - AutoCallable option batch simulations, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float currTime = monthDt;

    for (int i = 1; i <= nBinaryOptions; i++){
        observationDates.push_back(currTime);
        payoffs.push_back(10.0f);
        barriers.push_back(1.20f);

        currTime += monthDt;
    }

    int nSimulations = pow(2, 24);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/AutoCallableOption/ExecutionTimeMultipleOptions/Output/" + filename + ".csv");
    myFile << "Type,Engine,batchSize,nSimulationsPerOption,timeElapsed[s]\n";

    AutoCallableOptionSerialCPU *optionSerialCPU = new AutoCallableOptionSerialCPU(&asset, 50.0f, observationDates, barriers, payoffs,
                                                &monteCarloParams);
    AutoCallableOptionGPU *optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);

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

        myFile << "AutoCallableCall" << sep << "SerialCPU" << sep << nBatch << sep << nSimulations << sep << batchTimeCPU << "\n";
        myFile << "AutoCallableCall" << sep << "GPU" << sep << nBatch << sep << nSimulations << sep << batchTimeGPU << "\n";
    }

    myFile.close();

}

void OptionPricingAnalysisFacade::autoCallablePriceDependenceBarriersConstant() {
    cout << "\n [Running] - AutoCallable option price dependency from barriers, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    int nSimulations = pow(2, 20);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/AutoCallableOption/PriceDependenceBarrierConstant/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,nBinaryOptions,constantBarrier,value,stdError, confidence1, confidence2,timeElapsed\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float payoff = 10.0f;
    float currTime = monthDt;

    AutoCallableOptionGPU *optionGPU = nullptr;

    float constantBarrier = 0.0f;
    while(constantBarrier < 5) {
        currTime = monthDt;

        observationDates.clear();
        payoffs.clear();
        barriers.clear();

        for (int i = 1; i <= nBinaryOptions; i++) {
            observationDates.push_back(currTime);
            payoffs.push_back(payoff * i);
            barriers.push_back(constantBarrier);

            currTime += monthDt;
        }

        constantBarrier += 0.25f;


        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();
        myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << nBinaryOptions << sep << constantBarrier << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::autoCallableErrorDependenceBarriersConstant() {
    cout << "\n [Running] - AutoCallable option error dependency from barriers constant, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    int nSimulations = pow(2, 23);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/AutoCallableOption/AutoCallableOptionStdDevDependencyBarrier/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,nBinaryOptions,constantBarrier,value,stdError, confidence1,confidence2,stdDev,timeElapsed\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float payoff = 10.0f;
    float currTime = monthDt;

    AutoCallableOptionGPU *optionGPU = nullptr;

    float constantBarrier = 0.0f;
    while(constantBarrier < 5) {
        currTime = monthDt;

        observationDates.clear();
        payoffs.clear();
        barriers.clear();

        for (int i = 1; i <= nBinaryOptions; i++) {
            observationDates.push_back(currTime);
            payoffs.push_back(payoff * i);
            barriers.push_back(constantBarrier);

            currTime += monthDt;
        }

        constantBarrier += 0.25f;

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();

        double stdDev = sqrt(nSimulations) * gpuResultC.getStdError();

        myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << nBinaryOptions << sep << constantBarrier << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1] << sep << stdDev
               << sep << gpuResultC.getTimeElapsed() << "\n";
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::autoCallablePriceDependenceBarriersLinearIncreasing() {
    cout << "\n [Running] - AutoCallable option price dependency from liner increasing barriers, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    int nSimulations = pow(2, 20);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/AutoCallableOption/PriceDependenceBarriersLinearIncreasing/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,nBinaryOptions,firstBarrier,linearIncreasing,barriersOffset,value,stdError,confidence1, confidence2,timeElapsed\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float payoff = 10.0f;
    float currTime = monthDt;
    float linearBarriers = 0.0f;
    float linearIncreasing = 0.1f;

    AutoCallableOptionGPU *optionGPU = nullptr;

    float barriersOffset = 0.0f;
    while(barriersOffset < 5) {
        currTime = monthDt;

        observationDates.clear();
        payoffs.clear();
        barriers.clear();

        for (int i = 1; i <= nBinaryOptions; i++) {
            observationDates.push_back(currTime);
            payoffs.push_back(payoff * i);
            barriers.push_back(barriersOffset + linearBarriers + ((i - 1) * linearIncreasing));

            currTime += monthDt;
        }

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();
        myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << nBinaryOptions << sep << linearBarriers
               << sep << linearIncreasing << sep << barriersOffset << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";


        barriersOffset += 0.2f;
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::autoCallablePriceDependenceBarriersLinearDecreasing() {
    cout << "\n [Running] - AutoCallable option price dependency from liner decreasing barriers, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    int nSimulations = pow(2, 20);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";
    std::ofstream myFile("AnalysisData/AutoCallableOption/PriceDependenceBarriersLinearDecreasing/Output/" + filename + ".csv");
    myFile << "Type,nSimulations,Engine,nBinaryOptions,firstBarrier,linearDecreasing,barriersOffset,value,stdError,confidence1, confidence2,timeElapsed\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int nBinaryOptions = 12;
    float monthDt = 1.0f/nBinaryOptions;
    float payoff = 10.0f;
    float currTime = monthDt;
    float linearBarriers = 5.0f;
    float linearDecreasing = -0.1f;

    AutoCallableOptionGPU *optionGPU = nullptr;

    float barriersOffset = 0.0f;
    while(barriersOffset < 8) {
        currTime = monthDt;

        observationDates.clear();
        payoffs.clear();
        barriers.clear();

        for (int i = 1; i <= nBinaryOptions; i++) {
            observationDates.push_back(currTime);
            payoffs.push_back(payoff * i);
            barriers.push_back(-barriersOffset + linearBarriers + ((i - 1) * linearDecreasing));

            currTime += monthDt;
        }

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();
        myFile << "AutoCallableCall" << sep << nSimulations << sep << "GPU" << sep << nBinaryOptions << sep << linearBarriers
               << sep << linearDecreasing << sep << -barriersOffset << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";


        barriersOffset += 0.2f;
    }

    myFile.close();
}

void OptionPricingAnalysisFacade::autoCallableErrorDependenceNObsDates() {
    cout << "\n [Running] - AutoCallable option error trend simulation varying n observation dates, please wait...\n";

    Asset asset(100.0f, 0.3f, 0.03f);
    AutoCallableOption *optionGPU;

    int nSimulations = pow(2, 20);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/AutoCallableOption/ErrorTrendSimulationVaryingNObsDates/Output/" + filename + ".csv");
    myFile << "Type,Engine,nSimulations,nObsDates,value,stdError,confidence1,confidence2,timeElapsed[s]\n";

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;
    int maxBinaryOptions = 365;
    float dt = 0.0f;
    float currTime = 0.0f;

    for (int i = 1; i <= maxBinaryOptions; i++){
        dt = 1.0f/i;
        currTime = 0.0f;

        observationDates.clear();
        payoffs.clear();
        barriers.clear();

        while(currTime <= 1.0){
            observationDates.push_back(currTime);
            payoffs.push_back(10.0f);
            barriers.push_back(1.2f);

            currTime += dt;
        }

        optionGPU = new AutoCallableOptionGPU(&asset, 50.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
        SimulationResult gpuResultC = optionGPU->callPayoff();

        myFile << "AutoCallableCall" << sep << "GPU" << sep << nSimulations << sep << i << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
               << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
               << sep << gpuResultC.getTimeElapsed() << "\n";
    }


    myFile.close();
}

void OptionPricingAnalysisFacade::autoCallableCorrispondenceBinaryOption() {
    cout << "\n [Running] - AutoCallable option corrispondence to binary option analytical, please wait...\n";

    string gpuName = ContextGPU::instance()->getDeviceProp().name;
    string date = DateUtils().getDate();
    string filename = gpuName + " " + date;
    string sep = ",";

    std::ofstream myFile("AnalysisData/AutoCallableOption/CorrispondenceBinaryOptions/Output/" + filename + ".csv");


    Asset asset(100.0f, 0.3f, 0.02f);
    float rebase = 50.0f;
    float payoff = 10.0f;
    float timeToMaturity = 0.25f;
    float barrier = 115.0f;

    BinaryOption *binaryOption = new BinaryOptionAnalytical(&asset, timeToMaturity, barrier, payoff, rebase);
    double actualPayoff = exp(-asset.getRiskFreeRate() * timeToMaturity) * payoff * ((Result) binaryOption->callPayoff()).getValue();

    int nSimulations = pow(2, 26);
    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid = ContextGPU().instance()->getOptimalBlocksPerGrid(threadsPerBlock, nSimulations);
    GPUParams gpuParams(threadsPerBlock, blocksPerGrid);
    MonteCarloParams monteCarloParams(nSimulations, CURAND_RNG_PSEUDO_MTGP32, 42ULL);

    BinaryOptionGPU *binaryOptionGpu = new BinaryOptionGPU(&asset, timeToMaturity, barrier, payoff, rebase, &monteCarloParams, &gpuParams);
    actualPayoff = exp(-asset.getRiskFreeRate() * timeToMaturity) * payoff * ((Result) binaryOptionGpu->callPayoff()).getValue();

    std::vector<float> observationDates;
    std::vector<float> payoffs;
    std::vector<float> barriers;

    observationDates.push_back(timeToMaturity);
    payoffs.push_back(payoff);
    barriers.push_back(barrier/asset.getSpotPrice());

    AutoCallableOption *autoCallableOption = new AutoCallableOptionGPU(&asset, 0.0f, observationDates, barriers, payoffs, &monteCarloParams, &gpuParams);
    SimulationResult result = autoCallableOption->callPayoff();

    AutoCallableOption *autoCallableOptionC = new AutoCallableOptionSerialCPU(&asset, 0.0f, observationDates, barriers, payoffs, &monteCarloParams);
    SimulationResult resultC = autoCallableOptionC->callPayoff();

    myFile << "Type,Engine,nObsDates,value,nSimulations,stdError,confidence1,confidence2,timeElapsed[s]\n";
    myFile << "AutoCallableCall" << sep << "GPU" << sep << 1 << sep << result.getValue() << sep << nSimulations << sep << result.getStdError()
           << sep << result.getConfidence()[0] << sep << result.getConfidence()[1]
           << sep << result.getTimeElapsed() << "\n";
    myFile << "AutoCallableCall" << sep << "CPU" << sep << 1 << sep << resultC.getValue() << sep << nSimulations << sep << resultC.getStdError()
           << sep << resultC.getConfidence()[0] << sep << resultC.getConfidence()[1]
           << sep << resultC.getTimeElapsed() << "\n";
    myFile << "BinaryOptionCall" << sep << "Analytical" << sep << 1 << sep << actualPayoff << "\n";

    myFile.close();

}



