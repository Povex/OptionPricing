//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_ASSET_CUH
#define OPTIONPRICING_ASSET_CUH

class Asset {
private:
    float spot_price;

    float volatility;

    float risk_free_rate;

public:
    Asset();

    Asset(float spotPrice, float volatility, float riskFreeRate);

    virtual ~Asset();

    float getSpotPrice() const;

    void setSpotPrice(float spotPrice);

    float getVolatility() const;

    void setVolatility(float volatility);

    float getRiskFreeRate() const;

    void setRiskFreeRate(float riskFreeRate);

};




#endif //OPTIONPRICING_ASSET_CUH
