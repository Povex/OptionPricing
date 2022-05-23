//
// Created by marco on 23/05/22.
//

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#include "OptionPricingAnalysisFacade.cuh"
#include "../Model/InputOutput/GPUParams.cuh"
#include "../Model/InputOutput/MonteCarloParams.cuh"
#include "../Model/InputOutput/Asset.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionAnalytical.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionSerialCPU.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionGPU.cuh"
#include "../Utils/DateUtils.cuh"

using namespace std;

void OptionPricingAnalysisFacade::executeAnalysis() {
    europeanOptionsAnalysis();
}

void OptionPricingAnalysisFacade::europeanOptionsAnalysis() {
    std::string fname = "AnalysisData/ReliabilityAnalysis/Input/data.csv";
    vector<vector<string>> content;
    vector<string> row;
    string line, word;

    fstream file (fname, ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while(getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    else
        cout<<"Could not open the file\n";


    float spotPrice, volatility, riskFreeRate, strikePrice, timeToMaturity;
    spotPrice =  std::stof(content[1][0]);
    volatility = std::stof(content[1][1]);
    riskFreeRate = std::stof(content[1][2]);
    strikePrice = std::stof(content[1][3]);
    timeToMaturity = std::stof(content[1][4]);
    file.close();

    GPUParams gpuParams(256);
    MonteCarloParams monteCarloParams(1e8, 0);
    Asset asset(spotPrice, volatility, riskFreeRate);

    EuropeanOption *optionAnalytical, *optionSerialCPU, *optionGPU;

    optionAnalytical = new EuropeanOptionAnalytical(&asset, strikePrice, timeToMaturity);
    SimulationResult analyticalResultC = optionAnalytical->callPayoff();
    SimulationResult analyticalResultP = optionAnalytical->putPayoff();

    optionSerialCPU = new EuropeanOptionSerialCPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
    SimulationResult serialCpuResultC = optionSerialCPU->callPayoff();
    SimulationResult serialCpuResultP = optionSerialCPU->putPayoff();

    optionGPU = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
    SimulationResult gpuResultC = optionGPU->callPayoff();
    SimulationResult gpuResultP = optionGPU->putPayoff();

    string sep = ",";
    std::ofstream myFile("AnalysisData/ReliabilityAnalysis/Output/data "+ DateUtils().getDate() + ".csv");
    myFile << "Type,Engine,value,stdError, confidence1, confidence2,timeElapsed\n";
    myFile << "EuropeanCall" << "Analytical" << sep << analyticalResultC.getValue() << "\n";
    myFile << "EuropeanCall" << "SerialCPU" << sep << serialCpuResultC.getValue() << sep << serialCpuResultC.getStdError()
           << sep << serialCpuResultC.getConfidence()[0] << sep << serialCpuResultC.getConfidence()[1]
           << sep << serialCpuResultC.getTimeElapsed() << "\n";
    myFile << "EuropeanCall" << "GPU" << sep << gpuResultC.getValue() << sep << gpuResultC.getStdError()
           << sep << gpuResultC.getConfidence()[0] << sep << gpuResultC.getConfidence()[1]
           << sep << gpuResultC.getTimeElapsed() << "\n";
    myFile << "EuropeanPut" << "Analytical" << sep << analyticalResultP.getValue() << "\n";
    myFile << "EuropeanPut" << "SerialCPU" << sep << serialCpuResultP.getValue() << sep << serialCpuResultP.getStdError()
           << sep << serialCpuResultP.getConfidence()[0] << sep << serialCpuResultP.getConfidence()[1]
           << sep << serialCpuResultP.getTimeElapsed() << "\n";
    myFile << "EuropeanPut" << "GPU" << sep << gpuResultP.getValue() << sep << gpuResultP.getStdError()
           << sep << gpuResultP.getConfidence()[0] << sep << gpuResultP.getConfidence()[1]
           << sep << gpuResultP.getTimeElapsed() << "\n";
    myFile.close();
}
