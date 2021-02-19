from enum import IntEnum

class AutoMLStage(IntEnum):
    PREPROCESSING = 1
    SUGGEST = 2
    TEST = 3
    OBSERVE = 4
    VERIFY = 5