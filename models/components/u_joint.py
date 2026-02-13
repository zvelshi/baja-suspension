# default 
from dataclasses import dataclass

# ours
from .cv_joint import CVJoint

@dataclass
class UJoint(CVJoint):
    """
    U-Joint.
    For a position solver, this is kinematically identical to a Fixed CV Joint.
    (Phase/Velocity ripples don't affect static position).
    """

    pass