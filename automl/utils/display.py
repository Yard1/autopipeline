import pandas as pd
import ipywidgets as ipw
from IPython import get_ipython
from IPython.display import display, HTML, clear_output, update_display
import plotly.graph_objs as go

class IPythonDisplay:
    def __init__(self, id) -> None:
        self.id = id

    def display(self, obj):
        display(obj, display_id=self.id)