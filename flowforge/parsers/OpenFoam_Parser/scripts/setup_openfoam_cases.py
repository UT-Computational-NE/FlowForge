#!/usr/bin/env python3
"""
Create OpenFOAM run folders from a case_matrix CSV.

What this script does
- Reads a CSV from: <project_root>/data/case_matrix/
- Chooses a dataset number and creates cases under:
    <project_root>/data/run_folder/<mat>/dataset_<id>/run_1, run_2, ...
- Copies the template from:
    <project_root>/templates/<mat>/temp_<SIMTYPE>
- Performs template replacements using your `template.replace(...)`

Dataset numbering rules (NO zero padding)
- If the CSV stem already starts with dataset_<id>_ (or dataset_<id>.*), use that <id>
  and FAIL if dataset_<id> already exists under run_folder/<mat>.
- Otherwise, scan run_folder/<mat> for existing dataset_<id> folders and choose the
  LOWEST available dataset id (starting at 1).
  Then rename the CSV to start with dataset_<id>_ and proceed.

What this script does NOT do
- It does not run OpenFOAM. Running is manual.

Expected CSV columns (case_matrix):
  T, U_gas_inlet, U_liquid, Angle, centroid, radius
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from flowforge.materials.Fluid import Hitec
from scipy.spatial.transform import Rotation as R

import template  # your existing template.replace module


# ======================================================
# Dataset helpers (NO zero padding)
# ======================================================
DATASET_PREFIX_RE = re.compile(r"^dataset_(\d+)(?:_|\.)(.*)$", re.IGNORECASE)
DATASET_DIR_RE = re.compile(r"^dataset_(\d+)(?:_.+)?$", re.IGNORECASE)

def extract_dataset_id_from_filename(stem: str) -> int | None:
    """
    If filename stem begins with dataset_<id>_... or dataset_<id>... return <id> else None.
    """
    m = DATASET_PREFIX_RE.match(stem)
    if not m:
        return None
    return int(m.group(1))

def find_used_dataset_ids(mat_run_root: Path) -> set[int]:
    """
    Looks at directories under run_folder/<mat> and collects ids from names like:
      dataset_1
      dataset_2
    Ignores anything else.
    """
    used: set[int] = set()
    if not mat_run_root.exists():
        return used
    for p in mat_run_root.iterdir():
        if not p.is_dir():
            continue
        m = DATASET_DIR_RE.match(p.name)
        if m:
            used.add(int(m.group(1)))
    return used

def lowest_available_dataset_id(mat_run_root: Path, start: int = 1, stop: int = 999999) -> int:
    used = find_used_dataset_ids(mat_run_root)
    for did in range(start, stop + 1):
        if did not in used:
            return did
    raise RuntimeError(f"No available dataset id in range [{start}, {stop}] for {mat_run_root}")

def ensure_csv_has_dataset_prefix(csv_path: Path, dataset_id: int) -> Path:
    """
    If csv_path already begins with dataset_<id>_, leave it.
    Otherwise rename it to dataset_<id>_<oldname>.csv and return new Path.
    """
    stem = csv_path.stem
    if extract_dataset_id_from_filename(stem) is not None:
        return csv_path

    new_name = f"dataset_{dataset_id}_{csv_path.name}"
    new_path = csv_path.with_name(new_name)
    if new_path.exists():
        raise FileExistsError(f"Cannot rename CSV; target already exists: {new_path}")

    csv_path.rename(new_path)
    return new_path

def assert_dataset_dir_available(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        raise FileExistsError(
            f"Dataset folder already exists (collision): {dataset_dir}\n"
            "Choose a different dataset number (rename CSV prefix) or delete/rename the existing dataset folder."
        )


# ======================================================
# NEW: case-matrix resolution helpers
# ======================================================
DATASET_TOKEN_RE = re.compile(r"^(?:dataset_)?(\d+)$", re.IGNORECASE)

def resolve_case_matrix_path(case_matrix_arg: str, case_matrix_dir: Path) -> Path:
    """
    Accepts:
      - full/relative path to CSV
      - CSV filename inside case_matrix_dir
      - dataset id: "4"
      - dataset name: "dataset_4"
    Behavior for dataset id/name:
      - searches case_matrix_dir for CSVs whose *stem* starts with "dataset_<id>_"
        OR is exactly "dataset_<id>"
      - if 0 matches -> error
      - if >1 matches -> error (ambiguous)
    """
    s = case_matrix_arg.strip()

    # 1) direct path (as-given)
    p = Path(s)
    if p.exists() and p.is_file():
        return p

    # 2) filename inside case_matrix_dir
    p2 = case_matrix_dir / s
    if p2.exists() and p2.is_file():
        return p2

    # 3) dataset shorthand: "4" or "dataset_4"
    m = DATASET_TOKEN_RE.match(s)
    if m:
        did = int(m.group(1))
        prefix = f"dataset_{did}"

        matches: list[Path] = []
        if case_matrix_dir.exists():
            for fp in case_matrix_dir.iterdir():
                if not fp.is_file() or fp.suffix.lower() != ".csv":
                    continue
                stem = fp.stem
                # allow "dataset_<id>" or "dataset_<id>_anything"
                if stem == prefix or stem.startswith(prefix + "_"):
                    matches.append(fp)

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No case-matrix CSV found for {prefix} in: {case_matrix_dir}\n"
                f"Expected something like '{prefix}_... .csv' or '{prefix}.csv'."
            )
        if len(matches) > 1:
            matches_str = "\n  - " + "\n  - ".join(str(x.name) for x in sorted(matches))
            raise RuntimeError(
                f"Ambiguous dataset selection for {prefix} in: {case_matrix_dir}\n"
                f"Multiple matching CSV files found:{matches_str}\n"
                "Please pass the exact CSV filename/path."
            )

        return matches[0]

    # 4) nothing worked
    raise FileNotFoundError(
        f"Case matrix CSV not found: {case_matrix_arg}\n"
        f"Tried:\n"
        f"  - direct path: {Path(case_matrix_arg)}\n"
        f"  - inside:      {case_matrix_dir / case_matrix_arg}\n"
        f"  - dataset id/name shorthand like '4' or 'dataset_4'\n"
    )


# ======================================================
# OpenFOAM helpers
# ======================================================
def gravity(theta_deg: float) -> np.ndarray:
    vec = np.array([0.0, 1.0, 0.0])
    rotation_matrix = R.from_euler("z", theta_deg, degrees=True).as_matrix()
    g_rotated = -9.81 * (rotation_matrix @ vec)
    return np.where(np.abs(g_rotated) < 1e-10, 0.0, g_rotated)

def thermo_phy_mat(T: float, fluid: str):
    cp = PropsSI("C", "T", T, "P", 101325, fluid)
    mu = PropsSI("V", "T", T, "P", 101325, fluid)
    k = PropsSI("L", "T", T, "P", 101325, fluid)
    h = PropsSI("H", "T", T, "P", 101325, fluid)  # J/kg
    rho = PropsSI("D", "T", T, "P", 101325, fluid)
    Pr = (cp * mu) / k
    return cp, mu, Pr, h, rho

def ensure_executable(p: Path) -> None:
    if p.exists():
        p.chmod(0o755)

def clean_dir_contents(path: Path) -> None:
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

def copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        dst = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser(
        description="Create OpenFOAM dataset_<id> folders from a case_matrix CSV (no execution)."
    )

    parser.add_argument(
        "--case-matrix",
        required=True,
        type=str,
        help=(
            "Case matrix selector:\n"
            "  - CSV filename in data/case_matrix\n"
            "  - full path to CSV\n"
            "  - dataset id like '4'\n"
            "  - dataset name like 'dataset_4'\n"
            "If dataset id/name matches multiple CSVs, the script errors."
        ),
    )

    parser.add_argument("--mat", type=str, default="hitec_argon",
                        help="Material folder name used under templates/<mat> and run_folder/<mat>.")
    parser.add_argument("--liquid", type=str, default="hitec",
                        help="Liquid name used for surface tension logic. Default: hitec")
    parser.add_argument("--gas", type=str, default="argon",
                        help="Gas name passed to CoolProp. Default: argon")

    parser.add_argument("--sim-type", choices=["laminar", "LES", "RANS"], default="LES",
                        help="Select which template folder to use: temp_laminar, temp_LES, temp_RANS.")

    parser.add_argument("--old-run", action="store_true",
                        help="Use old_run template paths (0.orig). Default: off (use 0/ paths).")
    parser.add_argument("--injection-source", action="store_true",
                        help="Also replace topoSetDict/fvOptions/setFieldsDict if they exist (optional).")

    parser.add_argument("--length", type=float, default=0.5,
                        help="Pipe length used for end_time estimate. Default: 0.5")
    parser.add_argument("--endtime-max", type=float, default=4.0,
                        help="Upper cap for end_time estimate. Default: 4.0")

    parser.add_argument("--dataset-start", type=int, default=1,
                        help="Lowest dataset id to consider when auto-picking. Default: 1")

    args = parser.parse_args()

    # --------------------------------------------------
    # Project root + standard folders
    # scripts/<this_file>.py -> project_root
    # --------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    templates_root = project_root / "templates"
    data_root = project_root / "data"
    case_matrix_dir = data_root / "case_matrix"
    run_root = data_root / "run_folder"

    for d in [templates_root, data_root, case_matrix_dir, run_root]:
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Resolve case matrix CSV (UPDATED)
    # --------------------------------------------------
    csv_path = resolve_case_matrix_path(args.case_matrix, case_matrix_dir)

    # --------------------------------------------------
    # Template resolution (templates/<mat>/temp_LES etc)
    # --------------------------------------------------
    if args.sim_type.lower() == "les":
        template_dirname = "temp_LES"
    elif args.sim_type.lower() == "rans":
        template_dirname = "temp_RANS"
    elif args.sim_type.lower() == "laminar":
        template_dirname = "temp_laminar"
    else:
        raise ValueError(f"Unknown sim-type: {args.sim_type}")

    template_source = templates_root / args.mat / template_dirname
    if not template_source.exists():
        raise FileNotFoundError(f"Template not found: {template_source}")
    ensure_executable(template_source / "Allrun")

    # --------------------------------------------------
    # Decide dataset id and enforce uniqueness
    # --------------------------------------------------
    mat_run_root = run_root / args.mat
    mat_run_root.mkdir(parents=True, exist_ok=True)

    dataset_id = extract_dataset_id_from_filename(csv_path.stem)

    if dataset_id is not None:
        dataset_dir = mat_run_root / f"dataset_{dataset_id}"
        assert_dataset_dir_available(dataset_dir)
    else:
        dataset_id = lowest_available_dataset_id(mat_run_root, start=args.dataset_start)
        csv_path = ensure_csv_has_dataset_prefix(csv_path, dataset_id)

    dataset_dir = mat_run_root / f"dataset_{dataset_id}"
    assert_dataset_dir_available(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------
    # Load CSV (after potential rename)
    # --------------------------------------------------
    df = pd.read_csv(csv_path)

    required_cols = ["T", "U_gas_inlet", "U_liquid", "Angle", "centroid", "radius"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}. Expected at least: {required_cols}.")

    liquid = args.liquid
    gas = args.gas
    mat = args.mat

    print("Project root:    ", project_root)
    print("Case matrix CSV: ", csv_path)
    print("Template source: ", template_source)
    print("Dataset id:      ", dataset_id)
    print("Dataset output:  ", dataset_dir)

    # --------------------------------------------------
    # Create run folders (NON-padded run numbering)
    # --------------------------------------------------
    for i, row in df.iterrows():
        run_num = i + 1
        run_dir = dataset_dir / f"run_{run_num}"
        run_dir.mkdir(parents=True, exist_ok=True)

        clean_dir_contents(run_dir)
        copy_tree_contents(template_source, run_dir)
        ensure_executable(run_dir / "Allrun")

        # Extract parameters
        T = float(row["T"])
        theta = float(row["Angle"])
        centroid = float(row["centroid"])
        radius = float(row["radius"])
        U_gas = float(row["U_gas_inlet"])
        U_liq = float(row["U_liquid"])

        inequal = "<="

        g_xyz = gravity(theta)
        gas_cp, gas_mu, gas_pr, gas_enthalpy, gas_density = thermo_phy_mat(T, gas)

        if liquid == "hitec":
            liquid_surf = Hitec(mat).surface_tension(Hitec(liquid).enthalpy(T))
        else:
            liquid_surf = PropsSI("SURFACE_TENSION", "T", T, "Q", 0, liquid)

        end_time = round(min(max((3.0 * args.length) / U_liq, 0.5), args.endtime_max), 2)

        m_dot_gas = gas_density * (np.pi * (radius ** 2)) * U_gas
        e_dot_gas = m_dot_gas * gas_enthalpy

        replacements = {
            "gx": f"{g_xyz[0]:.4f}",
            "gy": f"{g_xyz[1]:.4f}",
            "gz": f"{g_xyz[2]:.4f}",
            "liquid_vel": f"{U_liq:.4f}",
            "liquid_temp": f"{T:.2f}",
            "liquid_surf": f"{liquid_surf:.6f}",
            "gas_vel": f"{U_gas:.4f}",
            "gas_temp": f"{T:.2f}",
            "gas_cp": f"{gas_cp:.2f}",
            "gas_mu": f"{gas_mu:.5e}",
            "gas_pr": f"{gas_pr:.4f}",
            "centroid": f"{centroid:.4f}",
            "radius": f"{radius:.4f}",
            "inequal": inequal,
            "time": end_time,
        }

        if args.injection_source:
            replacements.update({
                "m_dot_gas": f"{m_dot_gas:.5e}",
                "gas_enthalpy": f"{e_dot_gas:.2f}",
            })

        # Replace template variables
        if args.old_run:
            template.replace(template_source / "constant/g", run_dir / "constant/g", replacements)

            template.replace(template_source / f"0.orig/alpha.{gas}", run_dir / f"0.orig/alpha.{gas}", replacements)
            template.replace(template_source / f"0.orig/U.{liquid}", run_dir / f"0.orig/U.{liquid}", replacements)
            template.replace(template_source / f"0.orig/T.{liquid}", run_dir / f"0.orig/T.{liquid}", replacements)

            template.replace(template_source / "constant/phaseProperties", run_dir / "constant/phaseProperties", replacements)

            template.replace(template_source / f"0.orig/U.{gas}", run_dir / f"0.orig/U.{gas}", replacements)
            template.replace(template_source / f"0.orig/T.{gas}", run_dir / f"0.orig/T.{gas}", replacements)

            th = template_source / f"constant/thermophysicalProperties.{gas}"
            if th.exists():
                template.replace(th, run_dir / f"constant/thermophysicalProperties.{gas}", replacements)

            sfd = template_source / "system/setFieldsDict"
            if sfd.exists() and args.injection_source:
                template.replace(sfd, run_dir / "system/setFieldsDict", replacements)

            template.replace(template_source / "system/controlDict", run_dir / "system/controlDict", replacements)

        else:
            template.replace(template_source / f"0/alpha.{liquid}.orig", run_dir / f"0/alpha.{liquid}.orig", replacements)
            template.replace(template_source / f"0/U.{liquid}",           run_dir / f"0/U.{liquid}",           replacements)
            template.replace(template_source / f"0/T.{liquid}",           run_dir / f"0/T.{liquid}",           replacements)

            template.replace(template_source / "constant/g",               run_dir / "constant/g",               replacements)
            template.replace(template_source / "constant/phaseProperties", run_dir / "constant/phaseProperties", replacements)

            template.replace(template_source / f"0/alpha.{gas}.orig", run_dir / f"0/alpha.{gas}.orig", replacements)
            template.replace(template_source / f"0/U.{gas}",          run_dir / f"0/U.{gas}",          replacements)
            template.replace(template_source / f"0/T.{gas}",          run_dir / f"0/T.{gas}",          replacements)

            pp = template_source / f"constant/physicalProperties.{gas}"
            if pp.exists():
                template.replace(pp, run_dir / f"constant/physicalProperties.{gas}", replacements)

            if args.injection_source:
                topo = template_source / "system/topoSetDict"
                fvopt = template_source / "constant/fvOptions"
                sfd = template_source / "system/setFieldsDict"
                if topo.exists():
                    template.replace(topo, run_dir / "system/topoSetDict", replacements)
                if fvopt.exists():
                    template.replace(fvopt, run_dir / "constant/fvOptions", replacements)
                if sfd.exists():
                    template.replace(sfd, run_dir / "system/setFieldsDict", replacements)

            template.replace(template_source / "system/controlDict", run_dir / "system/controlDict", replacements)

    print(f"All {len(df)} simulations created in: {dataset_dir}")
    print("Run manually by executing ./Allrun inside each run_<n> directory.")


if __name__ == "__main__":
    main()
