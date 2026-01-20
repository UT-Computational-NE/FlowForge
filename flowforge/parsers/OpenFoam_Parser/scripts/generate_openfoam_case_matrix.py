#!/usr/bin/env python3
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc

# ============================================================
# Flow helpers (FINAL NAMING – lowercase)
# ============================================================
def readable_flow_name_from_angle(angle_is_ranged: bool, fixed_angle_value: float | None) -> str:
    if angle_is_ranged or fixed_angle_value is None:
        return "variable_flow"

    a = float(fixed_angle_value)
    if a == 0:
        return "vertical_upward"
    if 0 < a < 90:
        return "angled_upward"
    if a == 90:
        return "horizontal"
    if 90 < a < 180:
        return "angled_downward"
    if a == 180:
        return "vertical_downward"

    return "variable_flow"


# ============================================================
# Dataset helpers (NO zero padding)
# ============================================================
DATASET_FILE_RE = re.compile(r"^dataset_(\d+)_.*\.csv$", re.IGNORECASE)

def parse_existing_dataset_ids(outdir: Path) -> set[int]:
    used: set[int] = set()
    if not outdir.exists():
        return used

    for p in outdir.iterdir():
        if not p.is_file():
            continue
        m = DATASET_FILE_RE.match(p.name)
        if m:
            used.add(int(m.group(1)))
    return used


def lowest_available_dataset_id(outdir: Path, start: int = 1) -> int:
    used = parse_existing_dataset_ids(outdir)
    did = start
    while did in used:
        did += 1
    return did


def normalize_dataset_prefix(dataset_args: list[str] | None) -> str | None:
    if dataset_args is None:
        return None
    if len(dataset_args) not in (1, 2):
        raise ValueError("--dataset expects 1 value (id) or 2 values (id label).")
    try:
        did = int(dataset_args[0])
    except Exception:
        raise ValueError("--dataset id must be an integer.")
    if did < 1:
        raise ValueError("--dataset id must be >= 1.")
    return f"dataset_{did}"


def assert_dataset_id_unused(outdir: Path, dataset_prefix: str) -> None:
    did = int(dataset_prefix.split("_")[1])
    used = parse_existing_dataset_ids(outdir)
    if did in used:
        raise FileExistsError(
            f"Dataset id already exists in {outdir}: dataset_{did}"
        )


# ============================================================
# Parsing helpers
# ============================================================
def parse_range_or_single(values, default):
    if values is None:
        return False, float(default)
    if len(values) == 1:
        return False, float(values[0])
    if len(values) == 2:
        lo, hi = map(float, values)
        if hi <= lo:
            raise ValueError("Upper bound must be > lower bound.")
        return True, (lo, hi)
    raise ValueError("Provide 1 (fixed) or 2 (range) values only.")


# ============================================================
# Sampling helpers
# ============================================================
def generate_lhs(bounds, n, seed, rnd):
    if n <= 0:
        raise ValueError("--n must be > 0")
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    u = sampler.random(n)
    lower = np.array([b[0] for b in bounds.values()])
    upper = np.array([b[1] for b in bounds.values()])
    return pd.DataFrame(
        qmc.scale(u, lower, upper).round(rnd),
        columns=bounds.keys(),
    )


def generate_uniform(bounds, n, seed, rnd):
    if n <= 0:
        raise ValueError("--n must be > 0")
    rng = np.random.default_rng(seed)
    data = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in bounds.items()}
    return pd.DataFrame(data).round(rnd)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate case parameter matrix CSV")

    parser.add_argument("--mode", choices=["uniform", "lhs"], default="uniform")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--round", dest="round_decimals", type=int, default=4)

    parser.add_argument("--T", nargs="+", type=float)
    parser.add_argument("--ug", nargs="+", type=float)
    parser.add_argument("--ul", nargs="+", type=float)
    parser.add_argument("--angle", nargs="+", type=float)
    parser.add_argument("--centroid", nargs="+", type=float)
    parser.add_argument("--radius", nargs="+", type=float)

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Examples: '--dataset 7' or '--dataset new'")
    parser.add_argument("--short-name", action="store_true")

    args = parser.parse_args()

    DEFAULTS = {
        "T": 430.0,
        "U_gas_inlet": 0.3,
        "U_liquid": 0.03,
        "Angle": 0.0,
        "centroid": 0.0,
        "radius": 0.001,
    }

    parsed = {
        "T": parse_range_or_single(args.T, DEFAULTS["T"]),
        "U_gas_inlet": parse_range_or_single(args.ug, DEFAULTS["U_gas_inlet"]),
        "U_liquid": parse_range_or_single(args.ul, DEFAULTS["U_liquid"]),
        "Angle": parse_range_or_single(args.angle, DEFAULTS["Angle"]),
        "centroid": parse_range_or_single(args.centroid, DEFAULTS["centroid"]),
        "radius": parse_range_or_single(args.radius, DEFAULTS["radius"]),
    }

    bounds, fixed = {}, {}
    for k, (is_rng, val) in parsed.items():
        (bounds if is_rng else fixed)[k] = val

    parent = Path.cwd().parent
    outdir = Path(args.outdir) if args.outdir else parent / "data" / "case_matrix"
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset selection
    if args.dataset is None or (len(args.dataset) == 1 and args.dataset[0].lower() == "new"):
        did = lowest_available_dataset_id(outdir)
        dataset_prefix = f"dataset_{did}"
    else:
        dataset_prefix = normalize_dataset_prefix(args.dataset)
        assert_dataset_id_unused(outdir, dataset_prefix)

    # Sampling
    if bounds:
        df = generate_lhs(bounds, args.n, args.seed, args.round_decimals) if args.mode == "lhs" \
             else generate_uniform(bounds, args.n, args.seed, args.round_decimals)
        for k, v in fixed.items():
            df[k] = v
    else:
        df = pd.DataFrame([fixed] * args.n)

    df.insert(0, "case_id", np.arange(1, len(df) + 1))
    df = df[["case_id", "T", "U_gas_inlet", "U_liquid", "Angle", "centroid", "radius"]]

    flow_name = readable_flow_name_from_angle("Angle" in bounds, fixed.get("Angle"))
    filename = f"{dataset_prefix}_{flow_name}_{len(df)}_samples.csv"
    out_path = outdir / filename

    if out_path.exists():
        raise FileExistsError(f"Output CSV already exists: {out_path}")

    df.to_csv(out_path, index=False)

    print("Dataset:", dataset_prefix)
    print("Flow:   ", flow_name)
    print("Wrote:  ", out_path)
    print(df.head())


if __name__ == "__main__":
    main()
