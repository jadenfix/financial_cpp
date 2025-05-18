#pragma once
#include "Strategy.h"
#include "../core/Event.h"
#include "../core/EventQueue.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib> // For std::system
#include <deque>
#include <map>
#include <cmath>

// MLBridgeStrategy: C++/Python bridge for ML-based trading
class MLBridgeStrategy : public Strategy {
public:
    MLBridgeStrategy(const std::string& symbol, size_t lag = 5, size_t retrain_interval = 100)
        : symbol_(symbol), lag_(lag), retrain_interval_(retrain_interval), bar_count_(0) {}

    void handle_market_event(const MarketEvent& event, EventQueue& queue) override {
        auto it = event.data.find(symbol_);
        if (it == event.data.end()) return;
        const PriceBar& bar = it->second;
        double close = bar.Close;
        double volume = static_cast<double>(bar.Volume);
        // Update history
        close_history_.push_back(close);
        volume_history_.push_back(volume);
        if (close_history_.size() > lag_ + 1) close_history_.pop_front();
        if (volume_history_.size() > lag_ + 1) volume_history_.pop_front();
        // Only extract features if enough history
        if (close_history_.size() < lag_ + 1) return;
        // --- Feature extraction: lagged closes, returns, volume ---
        std::vector<double> features;
        for (size_t i = 0; i < lag_; ++i) {
            features.push_back(close_history_[close_history_.size() - 2 - i]);
            double ret = std::log(close_history_[close_history_.size() - 1 - i] / close_history_[close_history_.size() - 2 - i]);
            features.push_back(ret);
            features.push_back(volume_history_[volume_history_.size() - 2 - i]);
        }
        // Target: next return (if available)
        double target = std::log(close / close_history_[close_history_.size() - 2]);
        feature_buffer_.push_back({features, target});
        ++bar_count_;
        // --- Trigger training/prediction every retrain_interval_ bars ---
        if (bar_count_ % retrain_interval_ == 0) {
            export_features_to_csv("features.csv");
            // Call Python script for training/prediction
            std::system("python3 python_ml/train_predict.py --mode train --input features.csv --output predictions.csv");
            import_predictions_from_csv("predictions.csv", queue, event.timestamp);
            feature_buffer_.clear();
        }
    }

private:
    std::string symbol_;
    size_t lag_;
    size_t retrain_interval_;
    size_t bar_count_;
    std::deque<double> close_history_;
    std::deque<double> volume_history_;
    struct FeatureRow {
        std::vector<double> features;
        double target;
    };
    std::vector<FeatureRow> feature_buffer_;
    // --- Export features to CSV ---
    void export_features_to_csv(const std::string& filename) {
        std::ofstream ofs(filename);
        // Header
        for (size_t i = 0; i < lag_; ++i) {
            ofs << "lag_close_" << i << ",lag_ret_" << i << ",lag_vol_" << i << ",";
        }
        ofs << "target\n";
        for (const auto& row : feature_buffer_) {
            for (double f : row.features) ofs << f << ",";
            ofs << row.target << "\n";
        }
    }
    // --- Import predictions from CSV and generate signals ---
    void import_predictions_from_csv(const std::string& filename, EventQueue& queue, std::chrono::system_clock::time_point ts) {
        std::ifstream ifs(filename);
        std::string line;
        std::getline(ifs, line); // Skip header
        while (std::getline(ifs, line)) {
            double pred = std::stod(line);
            // Simple logic: LONG if pred > 0, SHORT if pred < 0
            if (pred > 0.0) {
                send_event(std::make_shared<SignalEvent>(ts, symbol_, SignalDirection::LONG), queue);
            } else if (pred < 0.0) {
                send_event(std::make_shared<SignalEvent>(ts, symbol_, SignalDirection::SHORT), queue);
            }
        }
    }
}; 