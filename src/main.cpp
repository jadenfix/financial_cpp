#include "core/Backtester.h"
#include "strategies/Strategy.h"
// --- Include ALL strategy headers ---
#include "strategies/MovingAverageCrossover.h"
#include "strategies/VWAPReversion.h"
#include "strategies/OpeningRangeBreakout.h"
#include "strategies/MomentumIgnition.h"
#include "strategies/PairsTrading.h"
#include "strategies/LeadLagStrategy.h"
#include "strategies/GBMStrategy.h"
#include "strategies/OrnsteinUhlenbeckStrategy.h"
#include "strategies/StochasticMLStrategy.h"
#include "strategies/BayesianLinearRegressionStrategy.h"
#include "strategies/GaussianProcessStrategy.h"
#include "strategies/GradientBoostedTreesStrategy.h"
#include "strategies/RandomForestStrategy.h"
#include "strategies/StackingStrategy.h"
#include "strategies/CrossValidationStrategy.h"
#include "strategies/MLBridgeStrategy.h"
#include "strategies/BayesianOnlineMLStrategy.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <map>
#include <iomanip>
#include <functional> // For std::function
#include <filesystem> // For checking data dir existence

// --- StrategyResult struct defined in Portfolio.h ---
#include "core/Portfolio.h" // Make sure this is included

// --- Helper Function to Build Data Path ---
std::string build_data_path(const std::string& base_dir, const std::string& subdir_name) {
    std::filesystem::path path = std::filesystem::path(base_dir) / subdir_name;
    return path.string();
}


int main(int argc, char* argv[]) {
    // --- UPDATED TITLE ---
    std::cout << "--- HFT Backtesting System - Comprehensive Multi-Strategy & Multi-Dataset Run ---" << std::endl;

    // --- Configuration ---
    std::string data_base_dir = "data";
    double initial_cash = 100000.0;

    // --- Define Datasets to Test ---
    std::vector<std::string> datasets_to_test = {
        "stocks_april",
        "2024_only",
        "2024_2025"
    };

    // --- Map to Store All Results ---
    // Key: Combined name like "StrategyName_on_DataSetName"
    std::map<std::string, StrategyResult> all_results;


    // --- OUTER LOOP: Iterate Through Datasets ---
    for (const std::string& target_dataset_subdir : datasets_to_test) {

        std::cout << "\n\n ///////////////////////////////////////////////////////////" << std::endl;
        std::cout << " ///// Starting Tests for Dataset: " << target_dataset_subdir << " /////" << std::endl;
        std::cout << " ///////////////////////////////////////////////////////////" << std::endl;


        // --- Build Final Data Path & Check Existence ---
        std::string data_path = build_data_path(data_base_dir, target_dataset_subdir);
        std::cout << "Using data path: " << data_path << std::endl;
        if (!std::filesystem::exists(data_path) || !std::filesystem::is_directory(data_path)) {
            std::cerr << "ERROR: Data directory '" << data_path << "' not found. Skipping dataset." << std::endl;
            continue; // Skip to the next dataset
        }

        // --- Define Symbol Names BASED ON CURRENT DATASET ---
        // Reset symbols for each dataset iteration
        std::string msft_sym = "", nvda_sym = "", goog_sym = "";
        std::string btc_sym = "", eth_sym = "", sol_sym = "", ada_sym = "";

        if (target_dataset_subdir == "stocks_april") {
            std::cout << "Loading symbols for dataset: stocks_april" << std::endl;
            msft_sym = "quant_seconds_data_MSFT";
            nvda_sym = "quant_seconds_data_NVDA";
            goog_sym = "quant_seconds_data_google";
        } else if (target_dataset_subdir == "2024_only") {
            std::cout << "Loading symbols for dataset: 2024_only" << std::endl;
            btc_sym = "btc_2024_data";
            eth_sym = "eth_2024_data";
            sol_sym = "sol_2024_data";
            ada_sym = "ada_2024_data";
        } else if (target_dataset_subdir == "2024_2025") {
            std::cout << "Loading symbols for dataset: 2024_2025" << std::endl;
            btc_sym = "2024_to_april_2025_btc_data";
            eth_sym = "2024_to_april_2025_eth_data";
            sol_sym = "2024_to_april_2025_solana_data"; // Match exact filename stem
            ada_sym = "2024_to_april_2025_ada_data";
        } else {
            std::cerr << "Warning: Unknown dataset subdirectory '" << target_dataset_subdir << "' encountered in loop logic." << std::endl;
            continue; // Skip dataset
        }

        // --- Define All Available Strategy Configurations FOR THIS DATASET ITERATION ---
        // Need to redefine inside loop because lambda captures [&] use current symbol values
        struct StrategyConfig {
            std::string name;
            std::function<std::unique_ptr<Strategy>()> factory;
            std::vector<std::string> required_datasets; // Datasets this config applies to
        };
        std::vector<StrategyConfig> available_strategies_this_iteration;

        // Define standard strategies (always applicable if dataset has OHLCV)
        available_strategies_this_iteration.push_back({"MACrossover_5_20", [](){ return std::make_unique<MovingAverageCrossover>(5, 20, 100.0); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"VWAP_2.0", [](){ return std::make_unique<VWAPReversion>(2.0, 100.0); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"ORB_30", [](){ return std::make_unique<OpeningRangeBreakout>(30, 100.0); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Momentum_5_10_2_3", [](){ return std::make_unique<MomentumIgnition>(5, 10, 2.0, 3, 100.0); }, {"stocks_april", "2024_only", "2024_2025"}});
        // --- New Strategies: Stochastic, ML, Bayesian, Ensemble ---
        available_strategies_this_iteration.push_back({"GBM", [&](){ return std::make_unique<GBMStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}}); // Example: use first symbol
        available_strategies_this_iteration.push_back({"OrnsteinUhlenbeck", [&](){ return std::make_unique<OrnsteinUhlenbeckStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"StochasticML", [&](){ return std::make_unique<StochasticMLStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"BayesianLinearRegression", [&](){ return std::make_unique<BayesianLinearRegressionStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"GaussianProcess", [&](){ return std::make_unique<GaussianProcessStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"GradientBoostedTrees", [&](){ return std::make_unique<GradientBoostedTreesStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"RandomForest", [&](){ return std::make_unique<RandomForestStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Stacking", [&](){ return std::make_unique<StackingStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"CrossValidation", [&](){ return std::make_unique<CrossValidationStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"MLBridge", [&](){ return std::make_unique<MLBridgeStrategy>(msft_sym.empty() ? btc_sym : msft_sym); }, {"stocks_april", "2024_only", "2024_2025"}});

        // Add our new Bayesian Online ML Strategy
        available_strategies_this_iteration.push_back({"BayesianOnlineML", [&](){ 
            return std::make_unique<BayesianOnlineMLStrategy>(
                msft_sym.empty() ? btc_sym : msft_sym,  // symbol
                50,                                      // feature_window
                1,                                       // update_interval (1 min)
                0.6,                                     // confidence_threshold
                100.0                                    // position_size
            ); 
        }, {"stocks_april", "2024_only", "2024_2025"}});

        // Define Pairs Trading - capture CURRENT symbols correctly using [&]
        double pairs_trade_value = 10000.0; size_t pairs_lookback = 60; double pairs_entry_z = 2.0, pairs_exit_z = 0.5;
        available_strategies_this_iteration.push_back({"Pairs_MSFT_NVDA", [&](){ return std::make_unique<PairsTrading>(msft_sym, nvda_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"stocks_april"}});
        available_strategies_this_iteration.push_back({"Pairs_NVDA_GOOG", [&](){ return std::make_unique<PairsTrading>(nvda_sym, goog_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"stocks_april"}});
        available_strategies_this_iteration.push_back({"Pairs_MSFT_GOOG", [&](){ return std::make_unique<PairsTrading>(msft_sym, goog_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"stocks_april"}});
        available_strategies_this_iteration.push_back({"Pairs_BTC_ETH", [&](){ return std::make_unique<PairsTrading>(btc_sym, eth_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Pairs_ETH_SOL", [&](){ return std::make_unique<PairsTrading>(eth_sym, sol_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Pairs_BTC_SOL", [&](){ return std::make_unique<PairsTrading>(btc_sym, sol_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Pairs_ETH_ADA", [&](){ return std::make_unique<PairsTrading>(eth_sym, ada_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"Pairs_SOL_ADA", [&](){ return std::make_unique<PairsTrading>(sol_sym, ada_sym, pairs_lookback, pairs_entry_z, pairs_exit_z, pairs_trade_value); }, {"2024_only", "2024_2025"}}); // Added missing SOL/ADA pair

        // Define Lead-Lag Strategies - capture CURRENT symbols
        size_t leadlag_window = 30; size_t leadlag_lag = 1; double leadlag_corr = 0.5, leadlag_ret = 0.0002, leadlag_size = 100.0;
        available_strategies_this_iteration.push_back({"LeadLag_MSFT->NVDA", [&](){ return std::make_unique<LeadLagStrategy>(msft_sym, nvda_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"stocks_april"}});
        available_strategies_this_iteration.push_back({"LeadLag_NVDA->MSFT", [&](){ return std::make_unique<LeadLagStrategy>(nvda_sym, msft_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"stocks_april"}});
        available_strategies_this_iteration.push_back({"LeadLag_BTC->ETH", [&](){ return std::make_unique<LeadLagStrategy>(btc_sym, eth_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"LeadLag_ETH->BTC", [&](){ return std::make_unique<LeadLagStrategy>(eth_sym, btc_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"LeadLag_ETH->SOL", [&](){ return std::make_unique<LeadLagStrategy>(eth_sym, sol_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"2024_only", "2024_2025"}});
        available_strategies_this_iteration.push_back({"LeadLag_SOL->ETH", [&](){ return std::make_unique<LeadLagStrategy>(sol_sym, eth_sym, leadlag_window, leadlag_lag, leadlag_corr, leadlag_ret, leadlag_size); }, {"2024_only", "2024_2025"}});


        // --- Filter strategies applicable to the CURRENT dataset ---
        std::vector<StrategyConfig> strategies_to_run_this_dataset;
        for (const auto& config : available_strategies_this_iteration) {
             bool applicable = false;
             for (const auto& ds_name : config.required_datasets) {
                 if (ds_name == target_dataset_subdir) {
                     applicable = true;
                     break;
                 }
             }
             if (applicable) {
                 strategies_to_run_this_dataset.push_back(config);
             }
        }

        if (strategies_to_run_this_dataset.empty()) {
            std::cout << "No applicable strategies found for dataset '" << target_dataset_subdir << "'. Skipping dataset." << std::endl;
            continue;
        }
         std::cout << "Preparing to run " << strategies_to_run_this_dataset.size() << " strategies for dataset '" << target_dataset_subdir << "'." << std::endl;


        // --- INNER LOOP: Iterate Through Applicable Strategies for this Dataset ---
        for (const auto& config : strategies_to_run_this_dataset) {
            std::cout << "\n\n===== Running Strategy: " << config.name << " on Dataset: " << target_dataset_subdir << " =====" << std::endl;

            std::unique_ptr<Strategy> strategy = nullptr;
            try {
                 strategy = config.factory(); // Create strategy using the factory
            } catch (const std::exception& e) {
                std::cerr << "Error creating strategy '" << config.name << "': " << e.what() << ". Skipping." << std::endl;
                continue; // Skip to next strategy
            }
            if (!strategy) { continue; } // Should not happen with factory, but safety check

            // Create a new Backtester for each specific run
            Backtester backtester(data_path, std::move(strategy), initial_cash);
            Portfolio const* result_portfolio = nullptr;

            try {
                result_portfolio = backtester.run_and_get_portfolio();
            } catch (const std::exception& e) {
                std::cerr << "FATAL ERROR during backtest for '" << config.name << "' on '" << target_dataset_subdir << "': " << e.what() << std::endl;
                continue; // Skip to next strategy
            } catch (...) {
                std::cerr << "FATAL UNKNOWN ERROR during backtest for '" << config.name << "' on '" << target_dataset_subdir << "'." << std::endl;
                continue; // Skip to next strategy
            }

            // Store results using a combined key
            if (result_portfolio) {
                std::string result_key = config.name + "_on_" + target_dataset_subdir;
                all_results[result_key] = result_portfolio->get_results_summary();
            } else {
                 std::cerr << "Warning: Backtest ran but portfolio pointer was null for " << config.name << " on " << target_dataset_subdir << "." << std::endl;
            }
            std::cout << "===== Finished Strategy: " << config.name << " on " << target_dataset_subdir << " =====" << std::endl;

        } // End INNER strategy loop

    } // End OUTER dataset loop


    // --- Print Combined Comparison Table ---
    if (!all_results.empty()) {
        std::cout << "\n\n===== COMBINED Strategy Comparison Results =====" << std::endl;
        std::cout << std::left << std::setw(50) << "Strategy (on Dataset)" // Wider column for combined name
                  << std::right << std::setw(15) << "Return (%)"
                  << std::right << std::setw(15) << "Max DD (%)"
                  << std::right << std::setw(15) << "Realized PnL"
                  << std::right << std::setw(15) << "Commission"
                  << std::right << std::setw(10) << "Fills"
                  << std::right << std::setw(18) << "Final Equity"
                  << std::endl;
        std::cout << std::string(138, '-') << std::endl; // Adjust separator width

        for (const auto& pair : all_results) {
            const std::string& name = pair.first; // Combined name
            const StrategyResult& res = pair.second;
            std::cout << std::left << std::setw(50) << name // Wider column
                      << std::fixed << std::setprecision(2)
                      << std::right << std::setw(15) << res.total_return_pct
                      << std::right << std::setw(15) << res.max_drawdown_pct
                      << std::right << std::setw(15) << res.realized_pnl
                      << std::right << std::setw(15) << res.total_commission
                      << std::right << std::setw(10) << res.num_fills
                      << std::right << std::setw(18) << res.final_equity
                      << std::endl;
        }
        std::cout << std::string(138, '=') << std::endl; // Adjust end separator
    } else {
         std::cout << "\nNo strategy results to display." << std::endl;
    }


    std::cout << "\n--- Comprehensive Run Invocation Complete ---" << std::endl;
    return 0;
}