//
// Created by marco on 02/05/22.
//

#ifndef OPTIONPRICING_ASSET_CUH
#define OPTIONPRICING_ASSET_CUH


class Asset {
public:
    float spot_price;
    float volatility;
    float risk_free_rate;

public:
    Asset();
    Asset(float spot_price, float volatility, float risk_free_rate);
};



#endif //OPTIONPRICING_ASSET_CUH
