#include "../ml_bridge.h"
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        MLBridge bridge(argv[1], argv[2], argv[3], "btc", false);
        double close = std::stod(argv[4]);
        auto [prediction, uncertainty] = bridge.getPredictionOHLCV(close, close + 100, close - 100, close, 1000);
        if (!std::isfinite(prediction) || !std::isfinite(uncertainty) || uncertainty < 0) return 3;
        std::cout << "BRIDGE_RESULT " << prediction << "," << uncertainty << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
