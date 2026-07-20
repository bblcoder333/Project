"""
summarize_emissions.py

Reads emissions log(s) produced by benchmark_pipeline.py — one CSV per model,
named outputs/emissions_log_<model_name>.csv (e.g. emissions_log_qwen.csv,
emissions_log_phi35.csv), since qwen and phi35 must be benchmarked from
separate venvs/processes and therefore write separate log files. This script
automatically finds and merges every emissions_log_*.csv file it can locate,
then collapses the combined data into a per-(model, agent) summary table with
mean/std for the metrics that matter for the report: energy (kWh), emissions
(kg CO2), and duration (s).

Also computes a per-(model, run) TOTAL row (summed across agents and screens)
so you can see whole-pipeline cost per run, not just per-agent.

Outputs:
  outputs/emissions_summary_by_agent.csv   -> mean/std per model x agent
  outputs/emissions_summary_by_run.csv     -> total per model x run (sanity check across the 5 runs)
  outputs/emissions_energy_by_agent.png    -> grouped bar chart, energy_kwh by agent, one color per model

Usage:
    python summarize_emissions.py
        (auto-discovers outputs/emissions_log_*.csv and merges them all)

    python summarize_emissions.py --log outputs/emissions_log_qwen.csv --log outputs/emissions_log_phi35.csv
        (explicitly list one or more log files instead of auto-discovering)

    python summarize_emissions.py --log-dir outputs/ --by-screen
        (search a different directory, and also write the per-screen breakdown)
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


METRICS = ["energy_kwh", "emissions_kg_co2", "duration_seconds"]

AGENT_ORDER = [
    "Agent1_Perception",
    "Agent2_Generation",
    "Agent3_Metamorphic",
    "Agent4_Optimization",
    "Agent5_Reduction",
]


def discover_logs(log_dir: str) -> list:
    """
    Find every emissions_log_*.csv in log_dir (one file per model, since
    each model's benchmark run writes its own file). Falls back to a plain
    outputs/emissions_log.csv if that's all that exists (e.g. an older run
    before the per-model split).
    """
    pattern = os.path.join(log_dir, "emissions_log_*.csv")
    found = sorted(glob.glob(pattern))
    if found:
        return found

    legacy = os.path.join(log_dir, "emissions_log.csv")
    if os.path.exists(legacy):
        return [legacy]

    return []


def load_logs(log_paths: list) -> pd.DataFrame:
    """Load and concatenate one or more emissions log CSVs into one DataFrame."""
    if not log_paths:
        sys.exit(
            "❌ No emissions log files found. Expected files like "
            "outputs/emissions_log_qwen.csv and outputs/emissions_log_phi35.csv "
            "— run benchmark_pipeline.py --model <name> first."
        )

    frames = []
    for path in log_paths:
        if not os.path.exists(path):
            sys.exit(f"❌ Could not find {path}.")
        df = pd.read_csv(path)
        missing = [c for c in ["model_name", "run_number", "screen_id", "agent"] + METRICS if c not in df.columns]
        if missing:
            sys.exit(f"❌ {path} is missing expected columns: {missing}")
        frames.append(df)
        print(f"   📄 {path} — {len(df)} rows, model(s): {sorted(df['model_name'].unique())}")

    combined = pd.concat(frames, ignore_index=True)

    # Sanity check: warn (don't fail) if the same model appears in more than
    # one file — usually means an old log wasn't cleaned up before a rerun.
    model_to_files = {}
    for path, df in zip(log_paths, frames):
        for m in df["model_name"].unique():
            model_to_files.setdefault(m, []).append(path)
    for m, files in model_to_files.items():
        if len(files) > 1:
            print(f"   ⚠️  Model '{m}' appears in multiple files: {files} — rows will be combined, "
                  f"double check this isn't stale/duplicate data.")

    return combined


def summarize_by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model_name, agent). Averages across ALL rows for that
    agent — i.e. across every screen AND every run — since the whole point
    of running 5 times is to get a stable mean +/- std per agent per model.
    """
    grouped = df.groupby(["model_name", "agent"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()

    # Order agents logically (Agent1 -> Agent4) instead of alphabetically
    grouped["_agent_order"] = grouped["agent"].apply(
        lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 99
    )
    grouped = grouped.sort_values(["model_name", "_agent_order"]).drop(columns="_agent_order")

    # Round for readability
    for col in grouped.columns:
        if col.endswith(("_mean", "_std")):
            grouped[col] = grouped[col].round(8)

    return grouped


def summarize_by_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model_name, screen_id, agent): mean/std across the 5 runs
    for that specific screen. Lets you check whether cost varies a lot by
    screen content (e.g. a screen with many test cases costing more in
    Agent 3) independent of which model was used.
    """
    grouped = df.groupby(["model_name", "screen_id", "agent"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()

    grouped["_agent_order"] = grouped["agent"].apply(
        lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 99
    )
    grouped = grouped.sort_values(["model_name", "screen_id", "_agent_order"]).drop(columns="_agent_order")

    for col in grouped.columns:
        if col.endswith(("_mean", "_std")):
            grouped[col] = grouped[col].round(8)

    return grouped


def summarize_by_run(df: pd.DataFrame) -> pd.DataFrame:

    """
    One row per (model_name, run_number): totals summed across every screen
    and every agent in that run. Useful as a sanity check that runs are
    consistent with each other (e.g. run 3 didn't spike due to thermal
    throttling or a partial failure).
    """
    totals = df.groupby(["model_name", "run_number"])[METRICS].sum().reset_index()
    totals = totals.sort_values(["model_name", "run_number"])
    for col in METRICS:
        totals[col] = totals[col].round(8)
    return totals


def plot_energy_by_agent(summary: pd.DataFrame, out_path: str):
    """Grouped bar chart: x-axis = agent, one bar cluster per model, y = mean energy_kwh with std error bars."""
    agents = [a for a in AGENT_ORDER if a in summary["agent"].unique()]
    models = sorted(summary["model_name"].unique())

    if not agents or not models:
        print("⚠️  Nothing to plot (empty summary).")
        return

    x = range(len(agents))
    bar_width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, model in enumerate(models):
        means, stds = [], []
        for agent in agents:
            row = summary[(summary["model_name"] == model) & (summary["agent"] == agent)]
            if row.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(row["energy_kwh_mean"].values[0])
                stds.append(row["energy_kwh_std"].values[0] if not pd.isna(row["energy_kwh_std"].values[0]) else 0)

        offsets = [xi + i * bar_width for xi in x]
        ax.bar(offsets, means, width=bar_width, yerr=stds, capsize=4, label=model)

    ax.set_xticks([xi + bar_width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels([a.replace("_", "\n") for a in agents], fontsize=9)
    ax.set_ylabel("Mean energy per screen (kWh)")
    ax.set_title("Energy consumption by agent and model (mean ± std across runs)")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize CodeCarbon emissions log(s) into per-model/per-agent stats.")
    parser.add_argument("--log", action="append", default=None,
                         help="Path to a specific emissions log CSV. Can be passed multiple times "
                              "(once per model). If omitted, auto-discovers outputs/emissions_log_*.csv.")
    parser.add_argument("--log-dir", default="outputs/",
                         help="Directory to auto-discover emissions_log_*.csv files in (default: outputs/)")
    parser.add_argument("--out", default="outputs/", help="Output directory for summary files")
    parser.add_argument("--by-screen", action="store_true",
                         help="Also write a per-(model, screen, agent) breakdown, "
                              "in addition to the default per-(model, agent) rollup.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    log_paths = args.log if args.log else discover_logs(args.log_dir)
    print(f"🔎 Using {len(log_paths)} log file(s):")
    df = load_logs(log_paths)

    print(f"\n✅ Loaded {len(df)} total rows")
    print(f"   Models: {sorted(df['model_name'].unique())}")
    print(f"   Runs per model: {df.groupby('model_name')['run_number'].nunique().to_dict()}")
    print(f"   Screens: {df['screen_id'].nunique()}")

    by_agent = summarize_by_agent(df)
    by_run = summarize_by_run(df)

    agent_path = os.path.join(args.out, "emissions_summary_by_agent.csv")
    run_path = os.path.join(args.out, "emissions_summary_by_run.csv")
    plot_path = os.path.join(args.out, "emissions_energy_by_agent.png")

    by_agent.to_csv(agent_path, index=False)
    by_run.to_csv(run_path, index=False)
    plot_energy_by_agent(by_agent, plot_path)

    print(f"\n💾 Per-agent summary  → {agent_path}")
    print(f"💾 Per-run totals     → {run_path}")
    print(f"📊 Chart              → {plot_path}")

    if args.by_screen:
        by_screen = summarize_by_screen(df)
        screen_path = os.path.join(args.out, "emissions_summary_by_screen.csv")
        by_screen.to_csv(screen_path, index=False)
        print(f"💾 Per-screen breakdown → {screen_path}")

    print("\n─── Per-agent summary (energy_kwh) ───")
    cols_to_show = ["model_name", "agent", "energy_kwh_mean", "energy_kwh_std",
                     "duration_seconds_mean", "emissions_kg_co2_mean"]
    print(by_agent[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()