#pragma once

#include "Event.h" // Includes PriceBar.h indirectly via Event.h using DataSnapshot
#include "Utils.h" // Include for formatting
#include <string>
#include <map>
#include <chrono>
#include <vector>
#include <numeric> // For std::accumulate
#include <iostream>
#include <iomanip> // For formatting output
#include <cmath>   // For std::abs
#include <limits>  // For std::numeric_limits
#include <algorithm> // For std::max, std::min

// --- Define StrategyResult struct here (can be used by main.cpp) ---
struct StrategyResult {
    double total_return_pct = 0.0;
    double max_drawdown_pct = 0.0;
    double realized_pnl = 0.0;
    double total_commission = 0.0;
    long num_fills = 0;
    double final_equity = 0.0;
    // Can add more metrics later (Sharpe, Sortino, Win Rate etc.)
};


class Portfolio {
public: // Make Position public so methods can access it easily

    struct Position {
        double quantity = 0.0;
        double average_price = 0.0;
        double market_value = 0.0;
        double unrealized_pnl = 0.0;
    };

private:
    double initial_cash_ = 100000.0;
    double current_cash_ = 0.0;
    std::map<std::string, Position> positions_;
    double total_commission_ = 0.0;
    double realized_pnl_ = 0.0;
    long num_fills_ = 0;
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> equity_curve_;

public:
    explicit Portfolio(double initial_cash = 100000.0)
        : initial_cash_(initial_cash), current_cash_(initial_cash) {}

    // --- handle_fill_event, update_market_values, record_equity ---
    // --- (same as previous correct version) ---
    void handle_fill_event(const FillEvent& event) {
        if (event.symbol.empty() || !std::isfinite(event.quantity) || event.quantity <= 0.0 ||
            !std::isfinite(event.fill_price) || event.fill_price <= 0.0 ||
            !std::isfinite(event.commission) || event.commission < 0.0) {
            std::cerr << "PORTFOLIO WARN: Ignoring invalid fill for '" << event.symbol << "'." << std::endl;
            return;
        }

        num_fills_++;
        total_commission_ += event.commission;
        Position& pos = positions_[event.symbol];
        const double previous_quantity = pos.quantity;
        const double previous_avg_price = pos.average_price;
        const double signed_delta =
            event.direction == OrderDirection::BUY ? event.quantity : -event.quantity;
        const double new_quantity = previous_quantity + signed_delta;
        constexpr double quantity_epsilon = 1e-9;

        const bool had_position = std::abs(previous_quantity) > quantity_epsilon;
        const bool same_direction = had_position &&
            ((previous_quantity > 0.0 && signed_delta > 0.0) ||
             (previous_quantity < 0.0 && signed_delta < 0.0));
        const bool opposing_direction = had_position && !same_direction;

        if (opposing_direction) {
            const double quantity_closed = std::min(std::abs(previous_quantity), std::abs(signed_delta));
            const double position_sign = previous_quantity > 0.0 ? 1.0 : -1.0;
            const double realized_for_this_trade =
                quantity_closed * (event.fill_price - previous_avg_price) * position_sign;
            realized_pnl_ += realized_for_this_trade;
            std::cout << "PORTFOLIO: Realized PnL on close: " << realized_for_this_trade << std::endl;
        }

        current_cash_ -= signed_delta * event.fill_price + event.commission;
        pos.quantity = std::abs(new_quantity) < quantity_epsilon ? 0.0 : new_quantity;

        if (pos.quantity == 0.0) {
            pos.average_price = 0.0;
        } else if (!had_position) {
            pos.average_price = event.fill_price;
        } else if (same_direction) {
            pos.average_price =
                (std::abs(previous_quantity) * previous_avg_price +
                 std::abs(signed_delta) * event.fill_price) /
                std::abs(pos.quantity);
        } else if ((previous_quantity > 0.0) == (pos.quantity > 0.0)) {
            // A partial close never changes the cost basis of the remaining position.
            pos.average_price = previous_avg_price;
        } else {
            // The fill crossed through flat. Only the residual opens at the fill price.
            pos.average_price = event.fill_price;
        }
        std::cout << "PORTFOLIO: Fill - Sym: " << event.symbol << ", Dir: " << (event.direction == OrderDirection::BUY ? "BUY" : "SELL") << ", Qty: " << event.quantity << " @ " << event.fill_price << ", NewPos: " << pos.quantity << ", AvgPx: " << pos.average_price << ", Cash: " << current_cash_ << std::endl;
        update_market_values({{event.symbol, PriceBar{event.timestamp, 0,0,0, event.fill_price, 0}}});
        record_equity(event.timestamp);
    }

    void update_market_values(const DataSnapshot& current_data) {
        for (auto& pair : positions_) {
            const std::string& symbol = pair.first;
            Position& pos = pair.second;
            if (std::abs(pos.quantity) < 1e-9) {
                pos.market_value = 0.0;
                pos.unrealized_pnl = 0.0;
                continue;
            }
            auto it = current_data.find(symbol);
            if (it != current_data.end()) {
                double current_price = it->second.Close;
                pos.market_value = pos.quantity * current_price;
                double cost_basis = pos.quantity * pos.average_price;
                pos.unrealized_pnl = pos.market_value - cost_basis;
            }
        }
    }

     void record_equity(const std::chrono::system_clock::time_point& timestamp) {
         double total_market_value = 0.0;
         for(const auto& pair : positions_) { total_market_value += pair.second.market_value; }
         double total_equity = current_cash_ + total_market_value;
         if (equity_curve_.empty() || equity_curve_.back().first != timestamp) {
              equity_curve_.emplace_back(timestamp, total_equity);
         } else {
              equity_curve_.back().second = total_equity;
         }
     }

    // --- Getters ---
    double get_current_cash() const { return current_cash_; }
    double get_total_equity() const {
         double total_market_value = 0.0;
         for(const auto& pair : positions_) { total_market_value += pair.second.market_value; }
        return current_cash_ + total_market_value;
    }
    const std::map<std::string, Position>& get_positions() const { return positions_; }
    const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& get_equity_curve() const { return equity_curve_; }
    double get_total_commission() const { return total_commission_; }
    double get_position_quantity(const std::string& symbol) const {
         auto it = positions_.find(symbol);
         return (it != positions_.end()) ? it->second.quantity : 0.0;
     }

    // --- Calculate and Print Performance Metrics ---
    void calculate_and_print_metrics() const {
        std::cout << "\n--- Performance Metrics ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        if (equity_curve_.empty()) {
            std::cout << "No equity data recorded." << std::endl; return;
        }
        double final_equity = get_total_equity();
        double total_return_pct = ((final_equity / initial_cash_) - 1.0) * 100.0;
        std::cout << "Ending Equity:       " << final_equity << std::endl;
        std::cout << "Total Return:        " << total_return_pct << "%" << std::endl;
        std::cout << "Realized PnL:        " << realized_pnl_ << " (Simplified)" << std::endl;
        std::cout << "Total Commission:    " << total_commission_ << std::endl;
        std::cout << "Total Fills/Trades:  " << num_fills_ << std::endl;

        double peak_equity = initial_cash_;
        double max_drawdown = 0.0;
        for (const auto& point : equity_curve_) {
            peak_equity = std::max(peak_equity, point.second);
            max_drawdown = std::max(max_drawdown, peak_equity - point.second);
        }
        double max_drawdown_pct = (peak_equity > 1e-9) ? (max_drawdown / peak_equity) * 100.0 : 0.0;
        std::cout << "Peak Equity Recorded: " << peak_equity << std::endl;
        std::cout << "Max Drawdown:        " << max_drawdown_pct << "%" << std::endl;
        std::cout << "--------------------------" << std::endl;
    }

    // --- Print Final Summary (includes metrics) ---
    void print_final_summary() const {
        std::cout << "\n--- Final Portfolio Summary ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Initial Cash:    " << initial_cash_ << std::endl;
        std::cout << "Ending Cash:     " << current_cash_ << std::endl;
         double total_market_value = 0.0;
         double total_unrealized_pnl = 0.0;
         for(const auto& pair : positions_) {
             total_market_value += pair.second.market_value;
             total_unrealized_pnl += pair.second.unrealized_pnl;
         }
        std::cout << "Market Value:    " << total_market_value << std::endl;
        std::cout << "Unrealized PnL:  " << total_unrealized_pnl << std::endl;
        std::cout << "Ending Positions:" << std::endl;
        bool has_positions = false;
        for (const auto& pair : positions_) {
            if (std::abs(pair.second.quantity) > 1e-9) {
                has_positions = true;
                std::cout << "  " << std::left << std::setw(30) << pair.first << ": "
                          << std::right << std::setw(10) << pair.second.quantity
                          << " @ Avg Cost " << std::setw(8) << pair.second.average_price
                          << " (MV: " << pair.second.market_value << ", UPL: " << pair.second.unrealized_pnl << ")"
                          << std::endl;
            }
        }
        if (!has_positions) { std::cout << "  (None)" << std::endl; }
        std::cout << "-----------------------------" << std::endl;
        calculate_and_print_metrics(); // Calls metrics calculation
    }

    // --- NEW: Method to return results struct ---
    StrategyResult get_results_summary() const {
        StrategyResult res;
        if (equity_curve_.empty()) {
            // If no equity points, return default struct but estimate final equity
            res.final_equity = get_total_equity(); // Based on final cash/MV
            res.total_return_pct = ((res.final_equity / initial_cash_) - 1.0) * 100.0;
            res.realized_pnl = realized_pnl_;
            res.total_commission = total_commission_;
            res.num_fills = num_fills_;
            res.max_drawdown_pct = 0.0; // Cannot calculate drawdown without curve
            return res;
        }

        res.final_equity = get_total_equity(); // Use getter for consistency
        res.total_return_pct = ((res.final_equity / initial_cash_) - 1.0) * 100.0;
        res.realized_pnl = realized_pnl_;
        res.total_commission = total_commission_;
        res.num_fills = num_fills_;

        // Calculate Max Drawdown from equity curve
        double peak_equity = initial_cash_;
        double max_drawdown = 0.0;
        for (const auto& point : equity_curve_) {
            peak_equity = std::max(peak_equity, point.second);
            max_drawdown = std::max(max_drawdown, peak_equity - point.second);
        }
        res.max_drawdown_pct = (peak_equity > 1e-9) ? (max_drawdown / peak_equity) * 100.0 : 0.0;

        return res;
    }
}; // End of Portfolio class definition
