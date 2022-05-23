//
// Created by marco on 23/05/22.
//

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "OptionPricingAnalysisFacade.cuh"
#include "../Model/Options/EuropeanOption/EuropeanOption.cuh"

using namespace std;

void OptionPricingAnalysisFacade::executeAnalysis() {
    europeanOptionsAnalysis();
}

void OptionPricingAnalysisFacade::europeanOptionsAnalysis() {
   /* std::string fname = "OptionsData/EuropeanOption/data.csv";
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
    vector<Result> results;
    float spotPrice, volatility, riskFreeRate, strikePrice, timeToMaturity;
    for(int i=1; i<2; i++)
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
        results.push_back(option->callPayoffSerialCPU());
        results.push_back(option->callPayoffBlackSholes());

        cout << results[0] << endl;
        cout << results[1] << endl;
        cout << results[2] << endl;
    }

    file.close();*/
}
