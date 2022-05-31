
#include "SharedAutoCallable.cuh"
#include "../../Shared/SharedFunctions.cuh"

/**
 * @brief   Calculates the payoff for one single sample in the path.
 *
 * @details This function generates the path with the geometric brownian motion.
 *          During the path generation, this function check all the observation dates,
 *          to check if the underling asset price is up or down the associated barrier.
 *          If the barrier is hitted than break the generation and return the actual payoff.
 *
 * @note    It assumes that al least one obseravtion dates exist (dateBarrierSize must be greater than zero),
 *          and the observationDates, barriers, payoffs are choerent with the problem.
 *
 * @param[in]   spotPrice           The start asset price (generally the actual asset price).
 * @param[in]   riskFreeRate        The risk free rate of the asset.
 * @param[in]   volatility          The volatility of the asset.
 * @param[in]   rebase              The rebase of the option if the barrier is never hitted.
 * @param[out]  samples             Array of samples generated.
 * @param[in]   normals             Array of samples of the standard gaussian normal.
 * @param[in]   observationDates    Array of dates to check.
 * @param[in]   barriers            Array of barriers to check.
 * @param[in]   payoffs             Array of payoffs.
 * @param[in]   dateBarrierSize     Size of the arrays: dates, barriers, payoffs.
 * @param[in]   i                   The index of the samples array to fill.
 *
 * @return      The actualized payoff of the first observation date hitted,
 *              or the actualized rebase if there isn't a barrier hit.
 */

__host__ __device__
void autoCallablePayoff(float spotPrice,
                        float riskFreeRate,
                        float volatility,
                        float rebase,
                        float *samples,
                        const float *normals,
                        const float *observationDates,
                        const float *barriers,
                        const float *payoffs,
                        int dateBarrierSize,
                        unsigned int i,
                        const int n_path){
    bool barrier_hit = false;
    float S = spotPrice;
    int date_index = 0;
    float dt = observationDates[date_index];

    while (date_index <= dateBarrierSize - 1) {
        S = generateS_T(spotPrice, riskFreeRate,
                        volatility, dt, normals[date_index * n_path + i]);

        if (S/spotPrice >= barriers[date_index]) { barrier_hit = true; break; }

        date_index++;
        dt = observationDates[date_index] - observationDates[date_index - 1];
    }

    if(!barrier_hit)
        samples[i] = exp(-riskFreeRate * observationDates[dateBarrierSize-1]) * rebase;
    else
        samples[i] = exp(-riskFreeRate * observationDates[date_index]) * payoffs[date_index];
}