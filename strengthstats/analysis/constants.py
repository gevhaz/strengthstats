from enum import Enum, auto


class ET(Enum):
    """Class holding names of exercise types."""

    TIME = auto()
    REPS = auto()
    WTIME = auto()
    WREPS = auto()
    OTHER = auto()
