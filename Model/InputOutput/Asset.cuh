//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_ASSET_CUH
#define OPTIONPRICING_ASSET_CUH

class Asset {
protected:
    float spot_price;

    float volatility;

    float risk_free_rate;

public:

    Asset(float spotPrice, float volatility, float riskFreeRate);

    ~Asset() = default;

    __host__ __device__ float getSpotPrice() const;

    void setSpotPrice(float spotPrice);

    __host__ __device__ float getVolatility() const;

    void setVolatility(float volatility);

    __host__ __device__ float getRiskFreeRate() const;

    void setRiskFreeRate(float riskFreeRate);

};




#endif //OPTIONPRICING_ASSET_CUH
