import pandas as pd
from ray.tune.callback import Callback


class BestPlotCallback(Callback):
    def __init__(self, widget, metric) -> None:
        self.widget = widget
        self.metric = metric
        self.best_results = []
        self.mean = []
        self.iterations = []
        super().__init__()

    def on_trial_result(self, iteration: int, trials, trial, result, **info):
        trial_score = None
        if result:
            trial_score = result[self.metric]

        if pd.isnull(trial_score):
            trial_score = self.mean[-1] if self.mean else 0

        if not self.iterations:
            self.iterations.append(1)
        else:
            self.iterations.append(self.iterations[-1] + 1)

        if not self.best_results or self.best_results[-1] < trial_score:
            self.best_results.append(trial_score)
        else:
            self.best_results.append(self.best_results[-1])

        if not self.mean:
            self.mean.append(trial_score)
        else:
            new_mean = self.mean[-1] + (trial_score - self.mean[-1]) / (
                len(self.mean) + 1
            )
            self.mean.append(new_mean)

        self.widget.data[0].x = self.iterations
        self.widget.data[1].x = self.iterations
        self.widget.data[0].y = self.best_results
        self.widget.data[1].y = self.mean

        self.widget.layout.title = f"Best {self.metric}={self.best_results[-1]}"
        self.widget.layout.xaxis.title = "Iterations"
        self.widget.layout.yaxis.title = self.metric

        lower_range_base = min(self.best_results[0], self.mean[-1])
        self.widget.layout.yaxis.range = [
            lower_range_base - (0.05 * lower_range_base),
            self.best_results[-1] + (0.05 * self.best_results[-1]),
        ]
