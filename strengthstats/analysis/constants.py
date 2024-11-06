from enum import Enum, auto


class ET(Enum):
    """Class holding names of exercise types."""

    TIME = auto()
    REPS = auto()
    WTIME = auto()
    WREPS = auto()
    OTHER = auto()


class Units:
    """What unit the volume is measured in for all exercise types.

    What unit the volume is measured in for all exercise types, both
    the abbreviation of the unit and the full name of the unit.
    """

    short: dict[ET, str] = {
        ET.TIME: "s",
        ET.REPS: "x",
        ET.WTIME: "kgs",
        ET.WREPS: "kg",
        ET.OTHER: "N/A",
    }
    long: dict[ET, str] = {
        ET.TIME: "seconds",
        ET.REPS: "times",
        ET.WTIME: "kilograms per second",
        ET.WREPS: "kilograms",
        ET.OTHER: "N/A",
    }
