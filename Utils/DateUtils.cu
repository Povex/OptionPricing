//
// Created by marco on 23/05/22.
//

#include <ctime>
#include "DateUtils.cuh"

using namespace std;

string DateUtils::getDate() {
    std::time_t t = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%c", std::localtime(&t));
    string date(buf);

    return date;
}
