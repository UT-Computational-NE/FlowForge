from .Solid import Graphite, SS316H
from .Fluid import FLiBe_UF4, Hitec

material_database = {
    # SOLIDS
    "graphite": Graphite,
    "ss316h": SS316H,

    # FLUIDS
    "flibe_uf4": FLiBe_UF4,
    "hitec": Hitec
}
