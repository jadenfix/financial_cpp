#include "core/ExecutionHandler.h"
#include "core/Portfolio.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr double kTolerance = 1e-8;

void expect_near(double actual, double expected, const std::string& label) {
    if (std::abs(actual - expected) > kTolerance) {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

const Portfolio::Position& position(const Portfolio& portfolio, const std::string& symbol) {
    const auto iterator = portfolio.get_positions().find(symbol);
    if (iterator == portfolio.get_positions().end()) {
        std::cerr << "missing position for " << symbol << '\n';
        std::exit(1);
    }
    return iterator->second;
}

FillEvent fill(OrderDirection direction, double quantity, double price, double commission = 0.0) {
    return FillEvent({}, "BTC", direction, quantity, price, commission);
}

void test_add_partial_and_full_close() {
    Portfolio portfolio;
    portfolio.handle_fill_event(fill(OrderDirection::BUY, 10.0, 100.0, 1.0));
    portfolio.handle_fill_event(fill(OrderDirection::BUY, 5.0, 120.0, 1.0));
    expect_near(position(portfolio, "BTC").quantity, 15.0, "increased long quantity");
    expect_near(position(portfolio, "BTC").average_price, 106.66666666666667, "weighted long basis");

    portfolio.handle_fill_event(fill(OrderDirection::SELL, 4.0, 130.0, 1.0));
    expect_near(position(portfolio, "BTC").quantity, 11.0, "partial close quantity");
    expect_near(position(portfolio, "BTC").average_price, 106.66666666666667, "partial close preserves basis");
    expect_near(portfolio.get_results_summary().realized_pnl, 93.33333333333333, "partial close realized pnl");

    portfolio.handle_fill_event(fill(OrderDirection::SELL, 11.0, 90.0, 1.0));
    expect_near(position(portfolio, "BTC").quantity, 0.0, "full close quantity");
    expect_near(position(portfolio, "BTC").average_price, 0.0, "full close clears basis");
    expect_near(portfolio.get_results_summary().realized_pnl, -90.0, "full close cumulative realized pnl");
    expect_near(portfolio.get_current_cash(), 99906.0, "cash reconciles with pnl and commissions");
}

void test_short_close_and_reversal() {
    Portfolio portfolio;
    portfolio.handle_fill_event(fill(OrderDirection::SELL, 5.0, 100.0, 0.5));
    portfolio.handle_fill_event(fill(OrderDirection::BUY, 8.0, 90.0, 0.5));
    expect_near(position(portfolio, "BTC").quantity, 3.0, "short-to-long reversal quantity");
    expect_near(position(portfolio, "BTC").average_price, 90.0, "reversal residual basis");
    expect_near(portfolio.get_results_summary().realized_pnl, 50.0, "short reversal realized pnl");

    portfolio.handle_fill_event(fill(OrderDirection::BUY, 2.0, 110.0));
    expect_near(position(portfolio, "BTC").quantity, 5.0, "increased reversal quantity");
    expect_near(position(portfolio, "BTC").average_price, 98.0, "weighted reversal basis");
}

void test_rejected_order_and_fill_are_immutable() {
    EventQueue queue;
    ExecutionHandler execution(queue);
    Portfolio portfolio;
    const auto timestamp = std::chrono::system_clock::time_point{};
    DataSnapshot snapshot;
    snapshot.emplace("BTC", PriceBar{timestamp, 100.0, 101.0, 99.0, 100.0, 10LL});
    MarketEvent market(timestamp, std::move(snapshot));

    execution.handle_order_event(
        OrderEvent(timestamp, "BTC", OrderType::MARKET, OrderDirection::BUY, -1.0), market);
    execution.handle_order_event(
        OrderEvent(timestamp, "BTC", OrderType::MARKET, OrderDirection::BUY,
                   std::numeric_limits<double>::quiet_NaN()), market);
    execution.handle_order_event(
        OrderEvent(timestamp, "MISSING", OrderType::MARKET, OrderDirection::BUY, 1.0), market);
    execution.handle_order_event(
        OrderEvent(timestamp, "BTC", OrderType::LIMIT, OrderDirection::BUY, 1.0), market);
    if (!queue.empty()) {
        std::cerr << "rejected order unexpectedly emitted a fill\n";
        std::exit(1);
    }

    portfolio.handle_fill_event(fill(OrderDirection::BUY, -1.0, 100.0, 1.0));
    expect_near(portfolio.get_current_cash(), 100000.0, "rejected fill cash");
    expect_near(portfolio.get_total_commission(), 0.0, "rejected fill commission");
    if (!portfolio.get_positions().empty() || portfolio.get_results_summary().num_fills != 0) {
        std::cerr << "rejected fill mutated portfolio state\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    test_add_partial_and_full_close();
    test_short_close_and_reversal();
    test_rejected_order_and_fill_are_immutable();
    std::cout << "PASS: signed-delta accounting transitions and rejection immutability\n";
}
