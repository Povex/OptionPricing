cmake_minimum_required(VERSION 3.22)
project(OptionPricing CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -ftz=true")

add_executable(OptionPricing main.cu Model/InputOutput/SimulationResult.cu Model/InputOutput/SimulationResult.cuh Model/InputOutput/Asset.cu Model/InputOutput/Asset.cuh Model/Options/DownOutOption/DownOutOption.cu Model/Options/DownOutOption/DownOutOption.cuh Model/Options/Option.cu Model/Options/Option.cuh Model/InputOutput/GPUParams.cu Model/InputOutput/GPUParams.cuh Model/InputOutput/MonteCarloParams.cu Model/InputOutput/MonteCarloParams.cuh Utils/errorHandler.cu Model/InputOutput/Result.cu Model/InputOutput/Result.cuh Model/StatisticUtils/Statistics.cu Model/StatisticUtils/Statistics.cuh Model/StatisticUtils/StatisticsSerialCPU.cu Model/StatisticUtils/StatisticsSerialCPU.cuh Model/StatisticUtils/StatisticsGPU.cu Model/StatisticUtils/StatisticsGPU.cuh Facades/OptionPricingFacade.cu Facades/OptionPricingFacade.cuh Facades/OptionPricingAnalysisFacade.cu Facades/OptionPricingAnalysisFacade.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionGPU.cuh Model/Options/EuropeanOption/EuropeanOption.cu Model/Options/EuropeanOption/EuropeanOption.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionAnalytical.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionSerialCPU.cuh Model/Options/Shared/SharedFunctions.cu Model/Options/Shared/SharedFunctions.cuh Utils/DateUtils.cu Utils/DateUtils.cuh Model/Options/AutocallableOption/AutoCallableOption.cu Model/Options/AutocallableOption/AutoCallableOption.cuh Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cu Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionGPU.cuh Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionSerialCPU.cu Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionSerialCPU.cuh Model/Options/AutocallableOption/Shared/SharedAutoCallable.cu Model/Options/AutocallableOption/Shared/SharedAutoCallable.cuh Utils/ContextGPU.cu Utils/ContextGPU.cuh Model/Options/BinaryOption/BinaryOption.cu Model/Options/BinaryOption/BinaryOption.cuh Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionAnalytical.cu Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionAnalytical.cuh Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionGPU.cu Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionGPU.cuh Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionCPU.cu Model/Options/BinaryOption/BinaryOptionImpl/BinaryOptionCPU.cuh Model/Options/BinaryOption/Shared/SharedBinaryOption.cu Model/Options/BinaryOption/Shared/SharedBinaryOption.cuh Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionCPU.cu Model/Options/EuropeanOption/EuropeanOptionImpl/EuropeanOptionCPU.cuh Model/StatisticUtils/StatisticsCPU.cu Model/StatisticUtils/StatisticsCPU.cuh Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cu Model/Options/AutocallableOption/AutoCallableOptionImpl/AutoCallableOptionCPU.cuh)

# https://stackoverflow.com/questions/24238937/how-to-change-a-compiler-flag-for-just-one-executable-in-cmake
target_link_libraries(OptionPricing -lcurand -lcuda -fopenmp)

# https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768/7
target_compile_options(OptionPricing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -fopenmp;
        >)

set_target_properties(OptionPricing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )








