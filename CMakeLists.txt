cmake_minimum_required(VERSION 3.22)
project(OptionPricing CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -ftz=true")

add_executable(OptionPricing main.cu Model/InputOutput/SimulationResult.cu Model/InputOutput/SimulationResult.cuh Model/InputOutput/Asset.cu Model/InputOutput/Asset.cuh Model/Options/DownOutOption/DownOutOption.cu Model/Options/DownOutOption/DownOutOption.cuh Model/Options/BinaryOption/BinaryOptionToDelete.cu Model/Options/BinaryOption/BinaryOptionToDelete.cuh Model/Options/Option.cu Model/Options/Option.cuh Model/InputOutput/GPUParams.cu Model/InputOutput/GPUParams.cuh Model/InputOutput/MonteCarloParams.cu Model/InputOutput/MonteCarloParams.cuh Utils/errorHandler.cu Model/InputOutput/Result.cu Model/InputOutput/Result.cuh Model/StatisticUtils/StatisticUtils.cu Model/StatisticUtils/StatisticUtils.cuh Model/StatisticUtils/StatisticUtilsCPU.cu Model/StatisticUtils/StatisticUtilsCPU.cuh Model/StatisticUtils/StatisticUtilsGPU.cu Model/StatisticUtils/StatisticUtilsGPU.cuh Facades/OptionPricingFacade.cu Facades/OptionPricingFacade.cuh Facades/OptionPricingAnalysisFacade.cu Facades/OptionPricingAnalysisFacade.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cuh Model/Options/EuropeanOption/EuropeanOption.cu Model/Options/EuropeanOption/EuropeanOption.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cuh Model/Options/Shared/SharedFunctions.cu Model/Options/Shared/SharedFunctions.cuh Utils/DateUtils.cu Utils/DateUtils.cuh Model/Options/AutocallableOption/AutoCallableOption.cu Model/Options/AutocallableOption/AutoCallableOption.cuh Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cu Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cuh Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cu Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cuh Model/Options/AutocallableOption/Shared/SharedAutoCallable.cu Model/Options/AutocallableOption/Shared/SharedAutoCallable.cuh Utils/ContextGPU.cu Utils/ContextGPU.cuh Model/Options/BinaryOption/BinaryOption.cu Model/Options/BinaryOption/BinaryOption.cuh Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionAnalytical.cu Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionAnalytical.cuh Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionGPU.cu Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionGPU.cuh)

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
target_link_libraries(OptionPricing -lcurand)

set_target_properties(OptionPricing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )








