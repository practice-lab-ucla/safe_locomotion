import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Step 1: Load results
# -------------------------
def load_results(results_root: str):
    """
    Walks through results_root/ENV_NAME/RUN_NAME and loads
    reward.npy and time_alive.npy into a nested dict:
      { ENV_NAME: { RUN_NAME: {'reward': np.ndarray, 'time_alive': np.ndarray} } }
    """
    results_root = Path(results_root)
    data = {}

    for env_dir in results_root.iterdir():
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        data[env_name] = {}

        for run_dir in env_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name

            reward_fp = run_dir / "reward.npy"
            time_alive_fp = run_dir / "time_alive.npy"

            results = {}
            if reward_fp.exists():
                results["reward"] = np.load(reward_fp)
            else:
                results["reward"] = None

            if time_alive_fp.exists():
                results["time_alive"] = np.load(time_alive_fp)
            else:
                results["time_alive"] = None

            data[env_name][run_name] = results
    return data


# -------------------------
# Step 2: Convert to DataFrame
# -------------------------
def build_reward_df(env_results: dict):
    """
    Given env_results = result_dict[env_name],
    build a DataFrame with columns = configs (alpha values, etc)
    and rows = trials.
    """
    reward_dict = {}
    for exp, res in env_results.items():
        values = res.get("reward", None)
        if values is None:
            continue
        # identify key
        if "warmup" in exp:
            key = exp.split("_")[0].replace("alpha", "")
            reward_dict[key] = np.sum(values, axis=1)
        elif "ppo" in exp:
            reward_dict[exp] = np.sum(values, axis=1)
    if not reward_dict:
        return None
    df = pd.DataFrame(reward_dict)
    # optionally drop specific columns
    if "0.02" in df.columns:
        df = df.drop("0.02", axis=1)
    return df


# -------------------------
# Step 3: Plotting functions
# -------------------------
def save_box_plot(df, cols, save_path, ylabel="reward"):
    means = df[cols].mean()
    sorted_cols = means.sort_values().index.tolist()
    data = [df[c] for c in sorted_cols]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=sorted_cols, showmeans=True, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.title("Reward Distribution (sorted by mean)")
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_cvar_plot(df, cols, save_path, alpha=0.3, ylabel="reward"):
    cvar = {}
    for c in cols:
        series = df[c].dropna()
        var = series.quantile(alpha)
        cvar[c] = series[series <= var].mean()

    sorted_cols = sorted(cvar, key=lambda c: cvar[c])
    sorted_vals = [cvar[c] for c in sorted_cols]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_cols, sorted_vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"CVaR (α={alpha}) by configuration")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_mean_plot(df, cols, save_path, ylabel="reward"):
    """
    Bar plot of mean cumulative reward with ±2·SE error bars.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are different configurations and whose rows
        are trial‑level cumulative rewards (already summed across time).
    cols : list-like
        Subset of df.columns to include in the plot.
    save_path : str or Path
        Where the PNG will be written.
    ylabel : str, optional
        y‑axis label, default is "reward".
    """
    # --- compute mean and 2·SE for each column ---
    means = df[cols].mean()
    std  = df[cols].std(ddof=1)          # unbiased sample std
    n    = df[cols].count()              # sample count (ignores NaNs)
    se   = std / np.sqrt(n)              # standard error of the mean
    err  = 2.0 * se                      # 2·SE as requested

    # --- sort columns by mean for readability ---
    sorted_cols = means.sort_values().index.tolist()
    sorted_means = means[sorted_cols]
    sorted_err   = err[sorted_cols]

    # --- plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_cols, sorted_means, yerr=sorted_err,
            capsize=4, alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title("Mean cumulative reward (error = ±2·SE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------
# Step 4: Main pipeline
# -------------------------
def process_all_results(results_root: str):
    all_results = load_results(results_root)
    for env_name, env_results in all_results.items():
        print(f"Processing environment: {env_name}")
        df = build_reward_df(env_results)
        if df is None or df.empty:
            print(f"  No valid reward data found for {env_name}. Skipping.")
            continue

        output_dir = os.path.join(results_root, env_name)
        os.makedirs(output_dir, exist_ok=True)

        box_plot_path = os.path.join(output_dir, "reward_box_plot.png")
        cvar_plot_path = os.path.join(output_dir, "reward_cvar_plot.png")
        mean_plot_path = os.path.join(output_dir, "reward_mean_plot.png")

        save_box_plot(df, df.columns, box_plot_path)
        save_cvar_plot(df, df.columns, cvar_plot_path)
        save_mean_plot(df, df.columns, mean_plot_path)

        print(f"  Saved plots for {env_name}:")
        print(f"    - {box_plot_path}")
        print(f"    - {cvar_plot_path}")
        print(f"    - {mean_plot_path}")


# -------------------------
# Step 5: Run
# -------------------------
if __name__ == "__main__":
    RESULTS_ROOT = "/home/danny/Documents/safe_locomotion/results"
    process_all_results(RESULTS_ROOT)
