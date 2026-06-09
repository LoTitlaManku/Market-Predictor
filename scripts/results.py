
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scripts.config import LEDGER_DIR, ROOT_DIR

########################################################################################################################

def collect_ledgers() -> pd.DataFrame:
    all_data = []
    # Get all validated rows of the ledgers
    for filename in tqdm(os.listdir(LEDGER_DIR.replace("ledgers", "ledgers")), desc="Analysing ledgers", unit="ledger"):
        try:
            filepath = os.path.join(LEDGER_DIR.replace("ledgers", "ledgers"), filename)
            ledger = pd.read_csv(filepath)

            completed = ledger.dropna(subset=['Actual_Price']).copy()
            completed["Ticker"] = filename.split("_")[0]
            if not completed.empty:
                all_data.append(completed)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    if not all_data:
        print("\nNo valid completed predictions found to analyze.")
        exit()

    # Combine everything into one analysis dataframe
    return pd.concat(all_data, ignore_index=True)

########################################################################################################################

def show_results(df: pd.DataFrame) -> None:
    total = len(df)
    model_cols = {
        "LGBM": "LGBM_signal",
        "SVC": "SVC_signal",
        "LASSO": "LASSO_signal",
        "LSTM": "LSTM_signal",
        "AVG": "AVG_signal"
    }

    print(f"{'=' * 50}\n--- OVERALL Performance Report ---")
    print(f"Total Evaluated: {total}")
    for name, col in model_cols.items():
        correct_column = f"{name}_correct"

        # Ensure types are correct for the whole dataframe first
        df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Market_Went_Up'] = (df["Actual_Price"] > df["Current_Price"]).astype(int) # noqa
        df[correct_column] = (( (df[col] > 0) & (df['Market_Went_Up'] == 1) ) |
                              ( (df[col] < 0) & (df['Market_Went_Up'] == 0) ) ).astype(int)

        dir_correct = df[correct_column].sum()
        print(f"\n[{name} Model Summary]")
        print(f"Total correct: {dir_correct}")
        print(f"Directional Accuracy: {(dir_correct / total) * 100:.2f}%")
        if name == "AVG":
            price_correct = (abs(df['Actual_Price'] - df['Predicted_Price']) / df['Actual_Price'] < 0.02).sum()
            print(f"Price Accuracy (2%): {(price_correct / total) * 100:.2f}%")

    horizons = ["1h", "2h", "4h", "25h", "1d", "2d", "7d", "28d"]
    for h in horizons:
        print(f"{'=' * 50}\n--- {h} Performance Report ---")
        # Filter for the specific horizon
        h_df = df[df['Horizon'] == h].copy()

        if h_df.empty:
            print(f"No data found for horizon: {h}")
            continue

        h_total = len(h_df)
        for name, col in model_cols.items():
            correct_column = f"{name}_correct"
            correct_h = h_df[correct_column].sum()

            print(f"\n[{name} Model Summary]")
            print(f"Total Predictions: {h_total}")
            print(f"Total correct: {correct_h}")
            print(f"Directional Accuracy: {(correct_h / h_total) * 100:.2f}%")

            # if name == "AVG":
            plot_signal_accuracy(h_df, h, col, correct_column)
            plot_signal_correctness(h_df, h, col, correct_column)

def plot_calibration_curve(df: pd.DataFrame, prob_col: str, horizon: str):
    df = df.copy()

    # Define edges and calculate midpoints for more points
    n_bins = 25
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Original Model Calibration
    df['prob_bin_orig'] = pd.cut(df[prob_col], bins=bin_edges)
    cal_orig = df.groupby('prob_bin_orig', observed=False)['Market_Went_Up'].mean()
    valid_orig = ~cal_orig.isna()

    # Flipped Model Calibration
    df['flipped_prob'] = 1 - df[prob_col]
    df['prob_bin_flip'] = pd.cut(df['flipped_prob'], bins=bin_edges)
    cal_flip = df.groupby('prob_bin_flip', observed=False)['Market_Went_Up'].mean()
    valid_flip = ~cal_flip.isna()

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Plot Original
    plt.plot(bin_centers[valid_orig], cal_orig.values[valid_orig], "o-",
             markersize=4, label=f"Original ({horizon})", alpha=0.6)

    # Plot Flipped
    plt.plot(bin_centers[valid_flip], cal_flip.values[valid_flip], "s-",
             markersize=4, label=f"Flipped ({horizon})", color='red')

    plt.xlabel("Predicted Probability (Confidence)")
    plt.ylabel("Actual Win Rate (How often price went UP)")
    plt.title(f"Calibration Curve: {prob_col} ({horizon})")
    plt.legend()
    plt.show()

def plot_signal_accuracy(h_df: pd.DataFrame, horizon: str, col_name: str, correct_col: str) -> None:
    # Create 20 bins from -1 to 1 (0.1 width each)
    bins = np.linspace(-1, 1, 101)
    # Use the center of each bin for the x-axis labels
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(12, 6))

    # Assign each row to a bin
    h_df['signal_bin'] = pd.cut(h_df[col_name], bins=bins, labels=bin_centers)

    # Group by bin and calculate mean accuracy and count
    stats = h_df.groupby('signal_bin', observed=False)[correct_col].agg(['mean', 'count'])

    # Filter out bins with too few samples to avoid statistical noise
    min_samples = 20
    valid_stats = stats[stats['count'] >= min_samples]

    if valid_stats.empty:
        print(f"Not enough data to plot signal accuracy for {horizon}")
        plt.close()
        return

    # Convert mean (0-1) to percentage (0-100)
    plt.plot(valid_stats.index, valid_stats['mean'] * 100,
             marker='o', linestyle='-', linewidth=2, label=f'{col_name} ({horizon}) Accuracy')

    # Reference line for random guessing
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')

    plt.title(f'Signal Strength vs. Accuracy')
    plt.xlabel('Signal Value')
    plt.ylabel('Directional Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()

    save_path = os.path.join(ROOT_DIR, "plots", f"{col_name}_{horizon}.png")
    plt.savefig(save_path)
    plt.close()

def plot_signal_correctness(h_df: pd.DataFrame, horizon: str, col_name: str, correct_col: str) -> None:
    # Use many bins to provide data for the smoothing function
    bins = np.linspace(-1, 1, 101)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Assign each row to a bin
    h_df = h_df.copy()
    h_df['signal_bin'] = pd.cut(h_df[col_name], bins=bins, labels=bin_centers)

    # Calculate total counts for correct and incorrect predictions per bin
    # Use observed=False to ensure we have a continuous index for smoothing
    stats = h_df.groupby('signal_bin', observed=False).apply(
        lambda x: pd.Series({
            'correct': (x[correct_col] == 1).sum(),
            'wrong': (x[correct_col] == 0).sum()
        })
    )

    # Apply Gaussian smoothing to the counts to get "smooth curves"
    # window=10 provides a balance between detail and smoothness
    smooth_stats = stats.rolling(window=10, win_type='gaussian', center=True).mean(std=3)

    plt.figure(figsize=(12, 6))

    # Plot the smoothed curves
    plt.plot(bin_centers, smooth_stats['correct'],
             color='green', linewidth=2.5, label='Total Correct (Smoothed)')
    plt.plot(bin_centers, smooth_stats['wrong'],
             color='red', linewidth=2.5, label='Total Incorrect (Smoothed)', alpha=0.7)

    # Fill areas to make the "Total" volume visible
    plt.fill_between(bin_centers, smooth_stats['correct'], color='green', alpha=0.1)
    plt.fill_between(bin_centers, smooth_stats['wrong'], color='red', alpha=0.05)

    plt.title(f'Prediction Frequency by Signal Strength ({horizon})')
    plt.xlabel('Signal Value (-1 to 1)')
    plt.ylabel('Smoothed Frequency (Total Predictions)')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()

    # Create the plot directory if it doesn't exist
    save_dir = os.path.join(ROOT_DIR, "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"signal_frequency_{horizon}_{col_name}.png")
    plt.savefig(save_path)
    plt.close()

########################################################################################################################

def find_top_predictable_tickers(df: pd.DataFrame) -> None:
    horizons = df['Horizon'].unique()

    # Ensure AVG_correct is calculated
    df['Market_Went_Up'] = (df["Actual_Price"] > df["Current_Price"]).astype(int) # noqa
    df['AVG_signal'] = pd.to_numeric(df['AVG_signal'], errors='coerce')
    df['AVG_correct'] = (((df['AVG_signal'] > 0) & (df['Market_Went_Up'] == 1)) |
                         ((df['AVG_signal'] < 0) & (df['Market_Went_Up'] == 0))).astype(int)

    print(f"\n{'=' * 60}")
    print(f"TOP TICKERS BY DIRECTIONAL ACCURACY")
    print(f"{'=' * 60}")

    for h in horizons:
        h_df = df[df['Horizon'] == h].copy()
        if h_df.empty: continue

        ticker_stats = h_df.groupby('Ticker')['AVG_correct'].agg(['mean', 'count']).reset_index()
        significant_tickers = ticker_stats[ticker_stats['count'] >= 50].copy()
        ticker_results = significant_tickers.sort_values(by='mean', ascending=False)
        top_50 = ticker_results.head(50)

        print(f"\n>>> Horizon: {h} (Found {len(significant_tickers)} significant tickers)")
        if top_50.empty:
            print("No tickers met the minimum prediction threshold.")
            continue

        for i, (idx, row) in enumerate(top_50.iterrows(), 1):
            print(f"{i:2}. {row['Ticker']:<6} | Accuracy: {row['mean'] * 100:6.2f}% | n={int(row['count'])}")

        ticker_results.to_csv(os.path.join(ROOT_DIR, "results", f"{h}.csv"), index=False)

        input(f'{"=" * 50} \n')

########################################################################################################################

def close_analysis(df, horizon, correct_column):
    # Filter specifically for the 1h horizon
    dfh = df[df['Horizon'] == horizon].copy()

    # Convert Open_Date to datetime and extract the hour
    dfh['Open_Date'] = pd.to_datetime(dfh['Open_Date'])
    dfh['Hour'] = dfh['Open_Date'].dt.hour

    # Group by hour to get accuracy percentage
    hourly_stats = dfh.groupby('Hour')[correct_column].agg(['mean', 'count']).reset_index()
    hourly_stats['mean'] *= 100  # Convert to percentage

    # Setup the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create the bar plot
    barplot = sns.barplot(data=hourly_stats, x='Hour', y='mean', palette="viridis")

    # Add a reference line at 50% (Random Guess) and your 26% (Current Average)
    plt.axhline(50, color='red', linestyle='--', label='Random (50%)')
    plt.axhline(dfh[correct_column].mean() * 100, color='blue', linestyle=':', label=f'{horizon} Average')

    # Labels and Formatting
    plt.title(f'{horizon} Prediction Accuracy by Time of Day (UTC)', fontsize=15)
    plt.ylabel('Directional Accuracy (%)')
    plt.xlabel('Hour of Day (24h Format)')
    plt.ylim(0, 100)
    plt.legend()

    # Add count labels on top of bars so you know if the data is "thin" for some hours
    for p in barplot.patches:
        hour_idx = int(p.get_x() + 0.5)
        if hour_idx < len(hourly_stats):
            count = hourly_stats.iloc[hour_idx]['count']
            barplot.annotate(f'n={int(count)}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'center',
                             xytext = (0, 9),
                             textcoords = 'offset points',
                             fontsize=9)

    plt.tight_layout()
    plt.show()

########################################################################################################################

if __name__ == '__main__':
    data = collect_ledgers()
    show_results(data)
    # find_top_predictable_tickers(data)
