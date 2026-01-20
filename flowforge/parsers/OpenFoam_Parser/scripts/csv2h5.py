import numpy as np
import pandas as pd
import h5py
import argparse
from pathlib import Path
from natsort import natsorted

# --------------------------------------------------
# Parse command-line arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_number", type=int, required=True, help="Run number extracted from the directory")
parser.add_argument("--material", type=str, required=True, help="Material name extracted from the directory")
parser.add_argument("--dataset_number", type=int, required=True, help="Dataset number extracted from the directory")
args = parser.parse_args()

run_num_list   = [args.run_number]
dataset_list   = [args.dataset_number]
material_list  = [args.material]

# --------------------------------------------------
# Resolve paths:
#   - current_dir: run directory (Allrun cd's here)
#   - script_dir:  openfoam_parser/scripts
#   - foam_base_dir: openfoam_parser
#   - data_root: openfoam_parser/data  (outputs land here)
# --------------------------------------------------
current_dir   = Path.cwd()
script_dir    = Path(__file__).resolve().parent
foam_base_dir = script_dir.parent
data_root     = foam_base_dir / "data"

for material in material_list:
    for dataset_num in dataset_list:
        for run_num in run_num_list:
            # NEW STANDARD (no padding)
            run_name = f"run_{run_num}"
            dataset_dirname = f"dataset_{dataset_num}"

            # Input: where data2csv.py wrote CSVs
            data_from_path = (
                data_root
                / "unprocessed"
                / material
                / dataset_dirname
                / run_name
            )

            # Output: where we write HDF5
            data_to_path = (
                data_root
                / "semiprocessed"
                / material
                / dataset_dirname
                / run_name
            )

            print(f"Processing {material} from {data_from_path}")
            data_to_path.mkdir(parents=True, exist_ok=True)

            if not data_from_path.exists():
                raise FileNotFoundError(f"Unprocessed data directory not found: {data_from_path}")

            # Get sorted time steps (subdirectories)
            time_list = natsorted(
                [p.name for p in data_from_path.iterdir() if p.is_dir()]
            )

            for t in time_list:
                print(t)
                data_folder_path = data_from_path / t

                # Parameter labels by material
                if material == "flibe_argon":
                    parameter_labels = [
                        "Area", "T.argon", "T.flibe", "p", "pMean", "p_rgh",
                        "U.argon:0", "U.argon:1", "U.argon:2",
                        "U.flibe:0", "U.flibe:1", "U.flibe:2",
                        "alpha.argon", "p",
                    ]
                elif material == "flibe_helium":
                    parameter_labels = [
                        "Area", "T.helium", "T.flibe", "p", "pMean", "p_rgh",
                        "U.helium:0", "U.helium:1", "U.helium:2",
                        "U.flibe:0", "U.flibe:1", "U.flibe:2",
                        "alpha.helium", "p",
                    ]
                elif material == "hitec_argon":
                    parameter_labels = [
                        "Area", "T.argon", "T.hitec", "p", "pMean", "p_rgh",
                        "U.argon:0", "U.argon:1", "U.argon:2",
                        "U.hitec:0", "U.hitec:1", "U.hitec:2",
                        "alpha.argon", "p",
                    ]
                elif material == "water_air":
                    parameter_labels = [
                        "Area", "T.air", "T.water", "p", "pMean", "p_rgh",
                        "U.air:0", "U.air:1", "U.air:2",
                        "U.water:0", "U.water:1", "U.water:2",
                        "alpha.air", "p",
                    ]
                elif material == "water_argon":
                    parameter_labels = [
                        "Area", "T.argon", "T.water", "p", "pMean", "p_rgh",
                        "U.argon:0", "U.argon:1", "U.argon:2",
                        "U.water:0", "U.water:1", "U.water:2",
                        "alpha.argon", "p",
                    ]
                else:
                    raise Exception(f"Unknown liquid-gas mixture name: {material}")

                points_labels = ["Points:0", "Points:1", "Points:2"]

                parameters = []
                xyz = []

                # Process CSV files at this time
                csv_files = natsorted(
                    [p for p in data_folder_path.iterdir() if p.suffix == ".csv"]
                )

                for csv_path in csv_files:
                    print(csv_path)
                    data = pd.read_csv(csv_path)

                    parameters.append(data[parameter_labels].to_numpy())
                    xyz.append(data[points_labels].to_numpy())

                    # Optionally remove CSVs after conversion:
                    # csv_path.unlink()

                parameters = np.array(parameters)
                xyz = np.array(xyz)

                # Save to HDF5
                output_file = (
                    data_to_path
                    / f"sliced_points_{t}.h5"
                )

                with h5py.File(output_file, "w") as hf:
                    hf.create_dataset("parameters", data=parameters)
                    hf.create_dataset("points", data=xyz)
                    hf.create_dataset("labels", data=parameter_labels)

            print(f"Processing complete for material {material}.")
            print(f"HDF5 files saved in {data_to_path}")
