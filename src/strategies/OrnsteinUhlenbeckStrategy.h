#pragma once
#include "Strategy.h"
#include "../core/Event.h"
#include "../core/EventQueue.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

// Ornstein-Uhlenbeck (OU) Process Strategy
class OrnsteinUhlenbeckStrategy : public Strategy {
public:
    OrnsteinUhlenbeckStrategy(const std::string& symbol, double kappa = 0.0, double theta = 0.0, double sigma = 0.0)
        : symbol_(symbol), kappa_(kappa), theta_(theta), sigma_(sigma) {}

    void handle_market_event(const MarketEvent& event, EventQueue& queue) override {
        // Find the symbol's price in the event data
        auto it = event.data.find(symbol_);
        if (it == event.data.end()) return;
        double price = it->second.Close;
        double log_price = std::log(price);
        log_price_history_.push_back(log_price);
        // Estimate parameters after enough data
        if (log_price_history_.size() > 20) {
            double mean = 0.0, var = 0.0;
            for (double x : log_price_history_) mean += x;
            mean /= log_price_history_.size();
            for (double x : log_price_history_) var += (x - mean) * (x - mean);
            var /= log_price_history_.size();
            theta_ = mean;
            sigma_ = std::sqrt(var);
            // Simple mean-reversion signal: buy if log_price < mean - threshold, sell if > mean + threshold
            double threshold = 0.5 * sigma_;
            if (log_price < mean - threshold) {
                send_event(std::make_shared<SignalEvent>(event.timestamp, symbol_, SignalDirection::LONG), queue);
            } else if (log_price > mean + threshold) {
                send_event(std::make_shared<SignalEvent>(event.timestamp, symbol_, SignalDirection::SHORT), queue);
            }
        }
    }

    // Optionally override handle_fill_event if needed

private:
    std::string symbol_;
    double kappa_;
    double theta_;
    double sigma_;
    std::vector<double> log_price_history_;
    // Add more state as needed
}; 