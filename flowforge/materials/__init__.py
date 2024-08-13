from .Solid import Graphite, SS316H
from .Fluid import FLiBe_UF4, Hitec

# SOLIDS
solid_material_database = {
    "graphite": Graphite,
    "ss316h": SS316H,
}

# FLUIDS
fluid_material_database = {
    "flibe_uf4": FLiBe_UF4,
    "hitec": Hitec
}
