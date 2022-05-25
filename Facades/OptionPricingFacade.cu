//
// Created by marco on 23/05/22.
//

#include "OptionPricingFacade.cuh"
#include "../Model/InputOutput/GPUParams.cuh"
#include "../Model/InputOutput/MonteCarloParams.cuh"
#include "../Model/InputOutput/Asset.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOption.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cuh"
#include "../Model/Options/AutocallableOption/AutoCallableOption.cuh"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

vector<SimulationResult> OptionPricingFacade::executeEuropeanCalls() {
    std::string fname = "OptionsData/EuropeanOption/data.csv";
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

    vector<EuropeanOption*> options(content.size());
    vector<SimulationResult> results;
    float spotPrice, volatility, riskFreeRate, strikePrice, timeToMaturity;
    for(int i=1; i<content.size(); i++)
    {
        spotPrice =  std::stof(content[i][0]);
        volatility = std::stof(content[i][1]);
        riskFreeRate = std::stof(content[i][2]);
        strikePrice = std::stof(content[i][3]);
        timeToMaturity = std::stof(content[i][4]);

        int nSimulations = 12e4;
        int threadsPerBlock = 512;
        int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
        GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

        MonteCarloParams monteCarloParams(12e4, 0);
        Asset asset(spotPrice, volatility, riskFreeRate);
        auto *option = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
        results.push_back(option->callPayoff());
    }

    file.close();

    return results;
}

vector<SimulationResult> OptionPricingFacade::executeAutoCallableCalls() {
    std::string fname = "OptionsData/AutoCallableOption/data.csv";
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

    vector<AutoCallableOption*> options(content.size());
    vector<SimulationResult> results;
    float spotPrice, volatility, riskFreeRate, strikePrice, timeToMaturity;
    for(int i=1; i<content.size(); i++)
    {
        spotPrice =  std::stof(content[i][0]);
        volatility = std::stof(content[i][1]);
        riskFreeRate = std::stof(content[i][2]);
        strikePrice = std::stof(content[i][3]);
        timeToMaturity = std::stof(content[i][4]);

        int nSimulations = 12e4;
        int threadsPerBlock = 512;
        int nBlocksPerGrid = ceil(float(nSimulations)/threadsPerBlock);
        GPUParams gpuParams((dim3(threadsPerBlock)), dim3(nBlocksPerGrid));

        MonteCarloParams monteCarloParams(nSimulations, 0);
        Asset asset(spotPrice, volatility, riskFreeRate);
        auto *option = new EuropeanOptionGPU(&asset, strikePrice, timeToMaturity, &monteCarloParams, &gpuParams);
        results.push_back(option->callPayoff());
    }

    file.close();

    return results;
}
