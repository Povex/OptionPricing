
#include "Model/Asset.cuh"
#include "Model/EuropeanOption.cuh"
#include "Model/SimulationResult.cuh"
#include "Model/BinaryOption.cuh"
#include "Model/AutocallableOption.cuh"

#include <iostream>
#include <time.h>
#include <bits/stdc++.h>
#include <vector>

using namespace std;

void testEuropeanOption(){
    Asset asset(100.0f, 0.3f, 0.01f);
    EuropeanOption option(asset, 100.0f, 1.0f);

    unsigned int n_simulations = 1e6;

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
    BinaryOption option(asset, 115.0f, 0.25f, 5.0f);

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

int main() {
    testAutoCallable();
}
