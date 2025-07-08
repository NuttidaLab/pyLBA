from dataclasses import dataclass, fields
import numpy as np

class Parameter:
    """Wraps a parameter value with type checking and conversion."""
    def __init__(self, name: str, value):
        if isinstance(value, (int, float)):
            self.value = float(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            self.value = np.array(value)
        else:
            raise TypeError(f"Parameter '{name}' must be numeric or array-like")
        self.name = name

    def __array__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return repr(self.value)

@dataclass(frozen=True)
class ModelParameters:
    def __post_init__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if not isinstance(val, Parameter):
                # bypass frozen to set the wrapped value
                object.__setattr__(
                    self,
                    f.name,
                    Parameter(f.name, val)
                )
                
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))})"