//
// Created by marco on 02/05/22.
//

#include "Asset.cuh"

Asset::Asset(float spot_price, float volatility, float risk_free_rate){
    this->spot_price = spot_price;
    this->volatility = volatility;
    this->risk_free_rate = risk_free_rate;
}

Asset::Asset(){}
