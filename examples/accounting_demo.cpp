#include "core/Portfolio.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void apply_and_report(Portfolio& portfolio, const std::string& label,
                      OrderDirection direction, double quantity, double price,
                      double commission = 1.0) {
    portfolio.handle_fill_event(FillEvent({}, "BTC", direction, quantity, price, commission));
    const auto& position = portfolio.get_positions().at("BTC");
    const auto results = portfolio.get_results_summary();
    std::cout << "RECEIPT," << label << ','
              << (direction == OrderDirection::BUY ? "BUY" : "SELL") << ','
              << quantity << ',' << price << ',' << position.quantity << ','
              << position.average_price << ',' << results.realized_pnl << ','
              << portfolio.get_current_cash() << ',' << results.total_commission << '\n';
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "label,direction,fill_quantity,fill_price,position,average_price,realized_pnl,cash,commission\n";

    Portfolio portfolio(100000.0);
    apply_and_report(portfolio, "open_long", OrderDirection::BUY, 10.0, 100.0);
    apply_and_report(portfolio, "increase_long", OrderDirection::BUY, 5.0, 120.0);
    apply_and_report(portfolio, "partial_close", OrderDirection::SELL, 4.0, 130.0);
    apply_and_report(portfolio, "full_close", OrderDirection::SELL, 11.0, 90.0);
    apply_and_report(portfolio, "open_short", OrderDirection::SELL, 5.0, 100.0);
    apply_and_report(portfolio, "reverse_long", OrderDirection::BUY, 8.0, 90.0);
}
