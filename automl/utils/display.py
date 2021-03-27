from abc import ABC, abstractmethod
import pandas as pd
import ipywidgets as ipw
from IPython import get_ipython
from IPython.display import display, HTML, clear_output, update_display
import plotly.graph_objs as go


class Display(ABC):
    @abstractmethod
    def __init__(self, id) -> None:
        pass

    @abstractmethod
    def display(self, obj):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def display_last_object(self):
        pass


class IPythonDisplay(Display):
    def __init__(self, id) -> None:
        self.id = id
        self._last_displayed_obj = None

    def display(self, obj):
        self._last_displayed_obj = obj
        display(obj, display_id=self.id)

    def clear(self):
        display(None, display_id=self.id)

    def display_last_object(self):
        display(self._last_displayed_obj, display_id=self.id)

    def clear_all(self):
        clear_output()