cmake_minimum_required(VERSION 3.12)
project(MlTradingBridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add the executable
add_executable(ml_bridge_test ml_bridge.cpp)

# Add threading library
find_package(Threads REQUIRED)
target_link_libraries(ml_bridge_test PRIVATE Threads::Threads)

# Output directory
set_target_properties(ml_bridge_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Print some info
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin") 