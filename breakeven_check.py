# breakeven_check.py
#
# Reads outputs/emissions_log.csv (real, measured Agent 5 generation cost)
# and outputs/energy_savings_summary.csv (projected per-cycle execution
# savings from Agent 5's tiering) after a main.py run, joins them by
# screen_id, and prints how many regression cycles it takes for Agent 5's
# own generation cost to pay for itself via the projected per-cycle savings.
#
# Usage:
#   python main.py
#   python breakeven_check.py

import csv
import os
import sys

EMISSIONS_LOG = "outputs/emissions_log.csv"
SAVINGS_SUMMARY = "outputs/energy_savings_summary.csv"


def load_agent5_generation_cost(path: str) -> dict:
    """Returns {screen_id: energy_kwh} for Agent5_Reduction rows only."""
    costs = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("agent") == "Agent5_Reduction":
                costs[row["screen_id"]] = float(row["energy_kwh"])
    return costs


def load_savings_summary(path: str) -> dict:
    """Returns {screen_id: row_dict} from energy_savings_summary.csv."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["screen_id"]] = row
    return rows


def main():
    if not os.path.exists(EMISSIONS_LOG):
        sys.exit(f"❌ {EMISSIONS_LOG} not found. Run `python main.py` first.")
    if not os.path.exists(SAVINGS_SUMMARY):
        sys.exit(f"❌ {SAVINGS_SUMMARY} not found. Run `python main.py` first.")

    generation_costs = load_agent5_generation_cost(EMISSIONS_LOG)
    savings_rows = load_savings_summary(SAVINGS_SUMMARY)

    if not generation_costs:
        sys.exit("❌ No Agent5_Reduction rows found in emissions_log.csv.")
    if not savings_rows:
        sys.exit("❌ No rows found in energy_savings_summary.csv.")

    screens = sorted(set(generation_costs.keys()) & set(savings_rows.keys()))
    missing = sorted(set(generation_costs.keys()) ^ set(savings_rows.keys()))
    if missing:
        print(f"⚠️  Screens present in only one file, skipping: {missing}")

    if not screens:
        sys.exit("❌ No matching screen_ids between the two files.")

    print(f"{'Screen':<25} {'Gen cost (kWh)':>15} {'Per-cycle savings (kWh)':>25} {'Break-even (cycles)':>20}")
    print("-" * 90)

    total_gen_cost = 0.0
    total_per_cycle_savings = 0.0

    for screen_id in screens:
        gen_cost = generation_costs[screen_id]
        row = savings_rows[screen_id]
        unoptimized = float(row["cost_per_cycle_unoptimized_kwh"])
        optimized = float(row["cost_per_cycle_optimized_kwh"])
        per_cycle_savings = unoptimized - optimized

        total_gen_cost += gen_cost
        total_per_cycle_savings += per_cycle_savings

        if per_cycle_savings > 0:
            breakeven = gen_cost / per_cycle_savings
            breakeven_str = f"{breakeven:.4f}"
        else:
            breakeven_str = "n/a (no savings)"

        print(f"{screen_id:<25} {gen_cost:>15.8f} {per_cycle_savings:>25.8f} {breakeven_str:>20}")

    print("-" * 90)
    if total_per_cycle_savings > 0:
        overall_breakeven = total_gen_cost / total_per_cycle_savings
        print(f"{'TOTAL':<25} {total_gen_cost:>15.8f} {total_per_cycle_savings:>25.8f} {overall_breakeven:>20.4f}")
    else:
        print(f"{'TOTAL':<25} {total_gen_cost:>15.8f} {total_per_cycle_savings:>25.8f} {'n/a':>20}")

    print("\nBreak-even (cycles) = how many regression cycles the tiered suite needs to run")
    print("before Agent 5's own generation cost is paid back by the per-cycle savings.")
    print("Lower is better; a value under 1.0 means it pays for itself before the first re-run.")


if __name__ == "__main__":
    main()