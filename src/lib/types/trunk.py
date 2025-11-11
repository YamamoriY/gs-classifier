from dataclasses import dataclass
import numpy as np

@dataclass
class TrunkLocation:
    trunk_locations: np.ndarray # (N, 3)