from enum import IntEnum

class AutoMLStage(IntEnum):
    PREPROCESSING = 1
    PRE_TUNE = 2
    TUNE = 3
    ENSEMBLE = 4
    VERIFY = 5