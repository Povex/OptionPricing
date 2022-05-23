cmake_minimum_required(VERSION 3.22)
project(OptionPricing CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -ftz=true")

add_executable(OptionPricing main.cu Model/InputOutput/SimulationResult.cu Model/InputOutput/SimulationResult.cuh Model/InputOutput/Asset.cu Model/InputOutput/Asset.cuh Model/Utilities/device_query.cu Model/Options/DownOutOption/DownOutOption.cu Model/Options/DownOutOption/DownOutOption.cuh Model/Options/AutocallableOption/AutocallableOption.cu Model/Options/AutocallableOption/AutocallableOption.cuh Model/Options/BinaryOption/BinaryOption.cu Model/Options/BinaryOption/BinaryOption.cuh Model/Utilities/OptionPricingUtils.cuh Model/Options/Option.cu Model/Options/Option.cuh Model/InputOutput/GPUParams.cu Model/InputOutput/GPUParams.cuh Model/InputOutput/MonteCarloParams.cu Model/InputOutput/MonteCarloParams.cuh Model/Utilities/errorHandler.cu Model/InputOutput/Result.cu Model/InputOutput/Result.cuh Model/Options/AutocallableOption/AutocallableOption2.cu Model/Options/AutocallableOption/AutocallableOption2.cuh Model/Utilities/StatisticUtils/StatisticUtils.cu Model/Utilities/StatisticUtils/StatisticUtils.cuh Model/Utilities/StatisticUtils/StatisticUtilsCPU.cu Model/Utilities/StatisticUtils/StatisticUtilsCPU.cuh Model/Utilities/StatisticUtils/StatisticUtilsGPU.cu Model/Utilities/StatisticUtils/StatisticUtilsGPU.cuh Facades/OptionPricingFacade.cu Facades/OptionPricingFacade.cuh Facades/OptionPricingAnalysisFacade.cu Facades/OptionPricingAnalysisFacade.cuh Model/Options/EuropeanOption/EuropeanOptionGPU.cu Model/Options/EuropeanOption/EuropeanOptionGPU.cuh Model/Options/EuropeanOption/EuropeanOption.cu Model/Options/EuropeanOption/EuropeanOption.cuh Model/Options/EuropeanOption/EuropeanOptionAnalytical.cu Model/Options/EuropeanOption/EuropeanOptionAnalytical.cuh Model/Options/EuropeanOption/EuropeanOptionSerialCPU.cu Model/Options/EuropeanOption/EuropeanOptionSerialCPU.cuh Model/Options/EuropeanOption/SharedFunctions.cu Model/Options/EuropeanOption/SharedFunctions.cuh Utils/DateUtils.cu Utils/DateUtils.cuh)

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
target_link_libraries(OptionPricing -lcurand)

set_target_properties(OptionPricing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )








