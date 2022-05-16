cmake_minimum_required(VERSION 3.22)
project(OptionPricing CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")

add_executable(OptionPricing main.cu Model/EuropeanOption.cu Model/EuropeanOption.cuh Model/SimulationResult.cu Model/SimulationResult.cuh Model/Asset.cu Model/Asset.cuh Utilities/device_query.cu Model/DownOutOption.cu Model/DownOutOption.cuh Model/AutocallableOption.cu Model/AutocallableOption.cuh Model/BinaryOption.cu Model/BinaryOption.cuh Utilities/OptionPricingUtils.cuh)

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
target_link_libraries(OptionPricing -lcurand)

set_target_properties(OptionPricing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )








