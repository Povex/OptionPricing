cmake_minimum_required(VERSION 3.22)
project(OptionPricing CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -ftz=true")

add_executable(OptionPricing main.cu Context/SimulationResult.cu Context/SimulationResult.cuh Context/Asset.cu Context/Asset.cuh Utilities/device_query.cu Model/DownOutOption/DownOutOption.cu Model/DownOutOption/DownOutOption.cuh Model/AutocallableOption/AutocallableOption.cu Model/AutocallableOption/AutocallableOption.cuh Model/BinaryOption/BinaryOption.cu Model/BinaryOption/BinaryOption.cuh Utilities/OptionPricingUtils.cuh Model/Option.cu Model/Option.cuh Context/GPUParams.cu Context/GPUParams.cuh Context/MonteCarloParams.cu Context/MonteCarloParams.cuh Model/EuropeanOption/EuropeanOption.cu Model/EuropeanOption/EuropeanOption.cuh Utilities/errorHandler.cu Context/Result.cu Context/Result.cuh Model/AutocallableOption/AutocallableOption2.cu Model/AutocallableOption/AutocallableOption2.cuh Utilities/StatisticUtils/StatisticUtils.cu Utilities/StatisticUtils/StatisticUtils.cuh Utilities/StatisticUtils/StatisticUtilsCPU.cu Utilities/StatisticUtils/StatisticUtilsCPU.cuh Utilities/StatisticUtils/StatisticUtilsGPU.cu Utilities/StatisticUtils/StatisticUtilsGPU.cuh Controllers/EuropeanCallController.cu Controllers/EuropeanCallController.cuh)

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
target_link_libraries(OptionPricing -lcurand)

set_target_properties(OptionPricing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )








