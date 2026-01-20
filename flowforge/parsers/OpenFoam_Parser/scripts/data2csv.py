from paraview.simple import *
from pathlib import Path
import os
import argparse

# Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# --------------------------------------------------
# Parse command-line arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_number", type=int, required=True,
    help="Run number extracted from the directory"
)
parser.add_argument(
    "--material", type=str, required=True,
    help="Material name extracted from the directory"
)
parser.add_argument(
    "--dataset_number", type=int, required=True,
    help="Dataset number extracted from the directory"
)
args = parser.parse_args()

run_num_list  = [args.run_number]
dataset_list  = [args.dataset_number]
material_list = [args.material]

# --------------------------------------------------
# Resolve paths  (UPDATED: script now lives in scripts/, outputs still in data/)
# --------------------------------------------------
current_dir   = Path.cwd()                        # .../data/run_folder/<mat>/dataset_<id>/run_<n>
script_dir    = Path(__file__).resolve().parent   # .../openfoam_parser/scripts
foam_base_dir = script_dir.parent                 # .../openfoam_parser (project root)
data_root     = foam_base_dir / "data"            # .../openfoam_parser/data

mesh_num = 1

for material in material_list:
    for dataset_num in dataset_list:
        for run_num in run_num_list:

            run_name = f"run_{run_num}"
            print(f"Processing {material} from {current_dir}")

            # --------------------------------------------------
            # OpenFOAM reader
            # --------------------------------------------------
            simulationfoam = OpenFOAMReader(FileName=str(current_dir / "output.foam"))
            simulationfoam.MeshRegions = ["internalMesh"]

            if material == "flibe_argon":
                simulationfoam.CellArrays = [
                    "T.argon", "T.flibe", "U", "U.argon", "U.argonMean",
                    "U.flibe", "U.flibeMean", "Ur",
                    "alpha.argon", "alpha.argonMean", "alpha.flibe",
                    "alphat.argon", "alphat.flibe",
                    "dgdt", "k.argon", "k.flibe",
                    "nut.argon", "nut.flibe",
                    "p", "pMean", "p_rgh"
                ]
            elif material == "hitec_argon":
                simulationfoam.CellArrays = [
                    "T.argon", "T.hitec", "U", "U.argon", "U.argonMean",
                    "U.hitec", "U.hitecMean", "Ur",
                    "alpha.argon", "alpha.argonMean", "alpha.hitec",
                    "alphat.argon", "alphat.hitec",
                    "dgdt", "k.argon", "k.hitec",
                    "nut.argon", "nut.hitec",
                    "p", "pMean", "p_rgh"
                ]
            elif material == "water_air":
                simulationfoam.CellArrays = [
                    "T.air", "T.water", "U", "U.air", "U.airMean",
                    "U.water", "U.waterMean", "Ur",
                    "alpha.air", "alpha.airMean", "alpha.water",
                    "alphat.air", "alphat.water",
                    "dgdt", "k.air", "k.water",
                    "nut.air", "nut.water",
                    "p", "pMean", "p_rgh"
                ]
            else:
                raise Exception(f"Unknown material: {material}")

            animationScene = GetAnimationScene()
            animationScene.UpdateAnimationUsingDataTimeSteps()

            # --------------------------------------------------
            # Time steps
            # --------------------------------------------------
            folders = [p.name for p in current_dir.iterdir() if p.is_dir()]
            numeric_folders = sorted([
                float(d) for d in folders
                if d.replace(".", "", 1).isdigit() and d not in ["0", "0.origin"]
            ])
            time_steps = [float(round(f, 3)) for f in numeric_folders]

            # --------------------------------------------------
            # Axial positions
            # --------------------------------------------------
            dz = 0.002
            L = 1.0 if mesh_num == 2 else 0.5
            N_y = int(L / dz)
            y_values = [round(dz * i, 3) for i in range(1, N_y + 1)]

            # --------------------------------------------------
            # Pipeline
            # --------------------------------------------------
            slice1 = Slice(Input=simulationfoam)
            slice1.SliceType = "Plane"
            slice1.SliceType.Normal = [0.0, 1.0, 0.0]

            generateSurfaceNormals = GenerateSurfaceNormals(Input=slice1)
            extractSurface = ExtractSurface(Input=generateSurfaceNormals)
            cellSize = CellSize(Input=extractSurface)
            cellSize.ComputeVertexCount = 0
            cellSize.ComputeLength = 0
            cellSize.ComputeVolume = 0

            cellDatatoPointData = CellDatatoPointData(Input=cellSize)
            cellDatatoPointData.PassCellData = 1
            cellDatatoPointData.PieceInvariant = 1

            cellCenters = CellCenters(Input=cellDatatoPointData)
            cellCenters.VertexCells = 1

            # --------------------------------------------------
            # Save CSVs
            #   data/unprocessed/<material>/dataset_<id>/run_<n>/<t>/
            # --------------------------------------------------
            for t in time_steps:
                animationScene.AnimationTime = t
                simulationfoam.UpdatePipeline(time=t)

                save_directory = (
                    data_root
                    / "unprocessed"
                    / material
                    / f"dataset_{dataset_num}"
                    / run_name
                    / f"{t}"
                )
                os.makedirs(save_directory, exist_ok=True)

                for idx, y in enumerate(y_values):
                    slice1.SliceType.Origin = [0.0, y, 0.0]
                    cellCenters.UpdatePipeline(time=t)
                    SaveData(str(save_directory / f"point_{idx}.csv"), proxy=cellCenters)

            print(
                f"CSV files saved under "
                f"{data_root}/unprocessed/{material}/dataset_{dataset_num}/{run_name}"
            )
