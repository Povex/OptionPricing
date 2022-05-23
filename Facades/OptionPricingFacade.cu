//
// Created by marco on 23/05/22.
//

#include "OptionPricingFacade.cuh"
#include "../Model/InputOutput/GPUParams.cuh"
#include "../Model/InputOutput/MonteCarloParams.cuh"
#include "../Model/InputOutput/Asset.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOption.cuh"

using namespace std;

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

vector<SimulationResult> OptionPricingFacade::executeEuropeanCall() {
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

        GPUParams gpuParams(256);
        MonteCarloParams monteCarloParams(12e4, 0);
        Asset asset(spotPrice, volatility, riskFreeRate);
        auto *option = new EuropeanOption(&asset, &gpuParams, &monteCarloParams, strikePrice, timeToMaturity);
        results.push_back(option->callPayoff());
    }

    file.close();

    return results;
}
