import pandas as pd
from abc import ABC


class EnsembleStrategy(ABC):
    def select_configurations(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile,
    ) -> list:
        return None


class RoundRobin(EnsembleStrategy):
    def select_configurations(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile,
    ) -> list:
        selected_configurations = []
        groupby_list = [
            f"config.{k}" for k in pipeline_blueprint.get_all_distributions().keys()
        ]
        groupby_list.reverse()
        grouped_results_df = results_df.sort_values(
            by="mean_test_score", ascending=False
        ).groupby(by=groupby_list)
        group_dfs = [
            group
            for name, group in grouped_results_df
            if group["mean_test_score"].max() >= percentile
        ]
        idx = 0
        iter = True
        while iter and any(len(group) > idx for group in group_dfs):
            for group in group_dfs:
                if len(selected_configurations) >= configurations_to_select:
                    iter = False
                    break
                if len(group) <= idx:
                    continue
                print(selected_configurations)
                print(idx)
                print(len(group))
                print(group.iloc[idx].name)
                selected_configurations.append(results[group.iloc[idx].name]["config"])
            idx += 1
        return selected_configurations


class RoundRobinEstimator(EnsembleStrategy):
    def select_configurations(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile,
    ) -> list:
        selected_configurations = []
        groupby_list = ["config.Estimator"]
        grouped_results_df = results_df.sort_values(
            by="mean_test_score", ascending=False
        ).groupby(by=groupby_list)
        group_dfs = [
            group
            for name, group in grouped_results_df
            if group["mean_test_score"].max() >= percentile
        ]
        idx = 0
        iter = True
        while iter and any(len(group) > idx for group in group_dfs):
            for group in group_dfs:
                if len(selected_configurations) >= configurations_to_select:
                    iter = False
                    break
                if len(group) <= idx:
                    continue
                print(selected_configurations)
                print(idx)
                print(len(group))
                print(group.iloc[idx].name)
                print(group.iloc[idx]["config.Estimator"])
                selected_configurations.append(results[group.iloc[idx].name]["config"])
            idx += 1
        return selected_configurations


class EnsembleBest(EnsembleStrategy):
    def select_configurations(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile,
    ) -> list:
        return None


class OneRoundRobinThenEnsembleBest(EnsembleStrategy):
    def select_configurations(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile,
    ) -> list:
        return None
