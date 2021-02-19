from typing import Any, Tuple, Union

def validate_type(var: Any, var_name:str, types: Union[type, Tuple[type]]):
    if not isinstance(var, types):
        raise TypeError(f"Expected {var_name} to be of type {types}, got {type(var)}")
