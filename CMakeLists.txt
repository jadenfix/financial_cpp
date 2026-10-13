# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.15) # Use 3.15 or higher for C++17 features

# Define the project name and language(s)
project(AlgoTradingSystem VERSION 1.0 LANGUAGES CXX)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Optional: Avoid compiler-specific extensions

# --- Compiler Warnings ---
# Add flags for common compilers. -Wno-unused-parameter suppresses warnings about unused function args.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
elseif(MSVC)
  # /W4 enables level 4 warnings, /permissive- enforces standard conformance more strictly
  add_compile_options(/W4 /permissive-)
endif()

# --- Find External Libraries (csv2) ---
# Define the path to the directory containing the 'csv2' subdirectory with reader.hpp
# Assumes 'external/csv2/include' exists directly under the project root where CMakeLists.txt resides.
set(CSV2_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/csv2/include")
# Check if the actual header file exists where expected and fail configuration if not found
if(NOT EXISTS "${CSV2_INCLUDE_DIR}/csv2/reader.hpp")
    message(FATAL_ERROR "csv2 header 'reader.hpp' not found at expected location: ${CSV2_INCLUDE_DIR}/csv2/\n"
                        "Please ensure the csv2 library is correctly placed in 'external/csv2/include/csv2/'")
endif()
message(STATUS "Found csv2 headers at: ${CSV2_INCLUDE_DIR}") # Print confirmation during configuration

# --- Define Source Files ---
# Group source files by component for better organization.
# List the .cpp files here. Header files (.h, .hpp) are found via include directories.
# If a component's implementation is entirely within its header file (header-only),
# you might not need a corresponding .cpp file listed here.

set(CORE_SOURCES
    # src/core/Portfolio.cpp          # Add if Portfolio class has separate implementation
    # src/core/ExecutionHandler.cpp   # Add if ExecutionHandler has separate implementation
    # src/core/Backtester.cpp         # Add if Backtester has separate implementation
    # src/core/Utils.cpp              # Add if Utils has separate implementation (likely header-only)
    # src/core/EventQueue.cpp         # Add if EventQueue has separate implementation (likely header-only)
)

set(DATA_SOURCES
    src/data/DataManager.cpp          # Implementation for data loading
    # src/data/PriceBar.cpp           # Add if PriceBar has separate implementation (likely header-only)
)

set(STRATEGY_SOURCES
    # Add .cpp files for strategies IF they have separate implementation files.
    # If strategy logic is purely in the .h file, don't list a .cpp here.
    # src/strategies/MovingAverageCrossover.cpp
    # src/strategies/VWAPReversion.cpp
    # src/strategies/OpeningRangeBreakout.cpp
    # src/strategies/MomentumIgnition.cpp
    # src/strategies/PairsTrading.cpp
    # src/strategies/Strategy.cpp      # Add if base Strategy class has implementation
)

# Define the main application entry point source file
set(MAIN_SOURCE src/main.cpp)


# --- Build Target ---
# Create the final executable target named 'trading_system'
# It will be built from all the specified source files.
add_executable(trading_system
    ${MAIN_SOURCE}
    ${CORE_SOURCES}
    ${DATA_SOURCES}
    ${STRATEGY_SOURCES}
    # Add other source groups here if needed (e.g., ${INDICATOR_SOURCES})
)

# --- Include Directories ---
# Tell the 'trading_system' target where to find header files during compilation.
target_include_directories(trading_system PRIVATE
    # Add the 'src' directory. This allows includes like #include "core/Event.h", etc.
    src
    # Add the directory containing the 'csv2' folder, as defined earlier.
    ${CSV2_INCLUDE_DIR}
)

# --- Link Libraries ---
# Link necessary system libraries.
# Link stdc++fs for <filesystem> support if needed (typically only older GCC < 9.0 on Linux)
# Modern compilers/standard libraries (like on macOS with Clang, or newer GCC/MSVC)
# often link <filesystem> automatically when C++17 standard is set.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(trading_system PRIVATE stdc++fs)
  message(STATUS "Linking against stdc++fs for older GCC on Linux")
endif()

# Add other required libraries here later using target_link_libraries()
# Example: target_link_libraries(trading_system PRIVATE SomeOtherLib::SomeOtherLib)


# --- Optional: Installation Rules ---
# These are commented out by default. Uncomment and adapt if you need to install the application.
# include(GNUInstallDirs)
# install(TARGETS trading_system
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )


# --- Optional: Set Output Directory ---
# By default, executables go into the top-level build directory.
# Uncomment to place the executable in a 'bin' subdirectory within the build folder.
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# --- Optional: Add Testing (using CTest) ---
# enable_testing()
# add_test(NAME MyTest COMMAND trading_system --run-tests) # Example if you add test mode


# --- Optional: Add Options ---
# Example: Allow user to build with debug symbols even in Release
# option(ENABLE_DEBUG_SYMBOLS_IN_RELEASE "Enable debug symbols in release builds" OFF)
# if(ENABLE_DEBUG_SYMBOLS_IN_RELEASE AND CMAKE_BUILD_TYPE STREQUAL "Release")
#   target_compile_options(trading_system PRIVATE -g) # Add debug symbols flag (-g for GCC/Clang)
# endif()