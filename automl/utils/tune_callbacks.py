import pandas as pd
import numpy as np
from ray.tune.callback import Callback


class BestPlotCallback(Callback):
    def __init__(self, widget, metric) -> None:
        self.widget = widget
        self.metric = metric
        self.best_results = []
        self.best_results_text = []
        self.best_results_test = []
        self.best_results_test_text = []
        self.results = []
        self.mean = []
        self.mean_text = []
        self.smallest_mean = 0
        self.iterations = []
        super().__init__()

    def on_trial_result(self, iteration: int, trials, trial, result, **info):
        trial_score = None
        text = result["trial_id"]
        if result:
            trial_score = result["metrics"][self.metric]
            if "test_metrics" in result:
                trial_test_score = result["test_metrics"][self.metric]
            else:
                trial_test_score = None

        if pd.isnull(trial_score):
            trial_score = self.mean[-1] if self.mean else None

        self.results.append(trial_score)

        if not self.iterations:
            self.iterations.append(1)
        else:
            self.iterations.append(self.iterations[-1] + 1)

        if not self.best_results or self.best_results[-1] < trial_score:
            self.best_results.append(trial_score)
            self.best_results_text.append(text)
        else:
            self.best_results.append(self.best_results[-1])
            self.best_results_text.append(self.best_results_text[-1])

        if trial_test_score is not None and (
            not self.best_results_test or self.best_results_test[-1] < trial_test_score
        ):
            self.best_results_test.append(trial_test_score)
            self.best_results_test_text.append(text)
        elif self.best_results_test:
            self.best_results_test.append(self.best_results_test[-1])
            self.best_results_test_text.append(self.best_results_test_text[-1])

        if not self.mean:
            self.mean.append(trial_score)
            self.smallest_mean = trial_score
        else:
            new_mean = self.mean[-1] + (trial_score - self.mean[-1]) / (
                len(self.mean) + 1
            )
            if new_mean < self.smallest_mean:
                self.smallest_mean = new_mean
            self.mean.append(new_mean)
        self.mean_text.append(text)

        self.widget.layout.title = f"Best {self.metric}={self.best_results[-1]}"
        self.widget.layout.xaxis.title = "Iterations"
        self.widget.layout.yaxis.title = self.metric

        lower_range_base = min(
            (self.best_results[0], self.best_results_test[0])
        )
        upper_range_base = max(
            (
                self.best_results[-1],
                self.best_results_test[-1] if self.best_results_test else -np.inf,
            )
        )
        self.widget.layout.yaxis.range = [
            lower_range_base - abs(0.05 * lower_range_base),
            upper_range_base + abs(0.05 * upper_range_base),
        ]

        self.widget.data[0].x = self.iterations
        self.widget.data[1].x = self.iterations
        self.widget.data[2].x = self.iterations
        self.widget.data[3].x = self.iterations
        self.widget.data[0].y = self.best_results
        self.widget.data[1].y = self.best_results_test
        self.widget.data[2].y = self.mean
        self.widget.data[3].y = self.results
        self.widget.data[0].text = self.best_results_text
        self.widget.data[1].text = self.best_results_test_text
        self.widget.data[2].text = self.mean_text
        self.widget.data[3].text = self.mean_text
