CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
PYTHON ?= python3
.PHONY: all test backtester
all: build/ml_bridge_test
build:
	mkdir -p build
build/ml_bridge_test: ml_bridge.cpp ml_bridge.h | build
	$(CXX) $(CXXFLAGS) ml_bridge.cpp -o $@
test:
	$(PYTHON) scripts/bridge-e2e.py
backtester:
	cmake -S . -B build/backtester
	cmake --build build/backtester
