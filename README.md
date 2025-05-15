/Users/jadenfix/algorithmic_trading_bots/
├── .git/                   # Hidden directory containing Git repository data
├── .gitignore              # Specifies intentionally untracked files (build/, *.o, etc.)
├── .vscode/                # (This should be in .gitignore but wasn't initially) - VS Code settings
├── CMakeLists.txt          # Instructions for CMake build system
├── data/                   # Directory containing input CSV data
│   ├── quant_seconds_data_MSFT.csv
│   ├── quant_seconds_data_NVDA.csv
│   └── quant_seconds_data_google.csv
├── external/
│   └── csv2                # <--- Currently a GITLINK (Mode 160000) referencing the csv2 repo commit
│                           #      Does NOT contain the actual csv2 code files in *your* repo yet.
└── src/                    # Directory containing C++ source code
    ├── DataManager.cpp     # Implementation of the DataManager class
    ├── DataManager.h       # Header declaration for the DataManager class
    ├── PriceBar.h          # Header definition for the PriceBar struct
    └── main.cpp            # Main executable entry point and demonstration logic

## Timestamp Handling: Extracting `date_only` and `time_only`

All scripts that load market data with a `timestamp` column automatically extract two additional columns:
- `date_only`: The date part of the timestamp (e.g., `2023-06-01`)
- `time_only`: The time part of the timestamp (e.g., `13:45:00`)

This is done using pandas after converting the `timestamp` column to datetime:

```python
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date_only'] = df['timestamp'].dt.date
    df['time_only'] = df['timestamp'].dt.time
    df.set_index('timestamp', inplace=True)
```

These columns are available for downstream analysis, C++ integration, and any custom logic that requires date or time granularity. Tests in `test_real_data.py` and `test_advanced_models.py` verify that these columns are present and correct after loading data.
