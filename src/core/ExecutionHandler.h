#pragma once

#include "Event.h"
#include "EventQueue.h"
#include "Utils.h" // Include for formatting
#include <string>
#include <map>
#include <memory>
#include <iostream> // For cout/cerr
#include <algorithm> // For std::max, std::abs
#include <cmath>

class ExecutionHandler {
private:
    EventQueue& event_queue_;

public:
    explicit ExecutionHandler(EventQueue& queue) : event_queue_(queue) {}

    void handle_order_event(const OrderEvent& order_event, const MarketEvent& next_market_event) {
        if (order_event.symbol.empty() || !std::isfinite(order_event.quantity) || order_event.quantity <= 0.0) {
            std::cerr << "SIM EXEC WARN: Rejected invalid order quantity for '"
                      << order_event.symbol << "'." << std::endl;
            return;
        }
        if (order_event.order_type == OrderType::MARKET) {
            auto symbol_iter = next_market_event.data.find(order_event.symbol);
            if (symbol_iter != next_market_event.data.end()) {
                const PriceBar& next_bar = symbol_iter->second;
                double fill_price = next_bar.Open;
                if (!std::isfinite(fill_price) || fill_price <= 0.0) {
                    std::cerr << "SIM EXEC WARN: Rejected invalid market price for "
                              << order_event.symbol << "." << std::endl;
                    return;
                }
                double commission = calculate_commission(order_event.quantity, fill_price);

                std::cout << "SIM EXEC: Order for " << order_event.quantity << " " << order_event.symbol
                          << (order_event.direction == OrderDirection::BUY ? " BUY" : " SELL")
                          << " filled at " << fill_price << std::endl;

                EventPtr fill_ev = std::make_shared<FillEvent>(
                    next_market_event.timestamp,
                    order_event.symbol,
                    order_event.direction,
                    order_event.quantity,
                    fill_price,
                    commission
                );
                event_queue_.push(std::move(fill_ev));
            } else {
                std::cerr << "SIM EXEC WARN: No market data for " << order_event.symbol
                          << " at " << formatTimestampUTC(next_market_event.timestamp) // Use formatter
                          << " to fill order." << std::endl;
            }
        } else {
             std::cerr << "SIM EXEC WARN: Limit orders not implemented." << std::endl;
        }
    }

private:
    // Example commission calculation
    double calculate_commission(double quantity, double price) {
        // Suppress warning if price is not used in this simple model
        (void)price; // Explicitly mark price as unused for now
        // Example: $0.005 per share, minimum $1.00
        double commission = std::abs(quantity) * 0.005;
        return std::max(1.0, commission);
    }
};
