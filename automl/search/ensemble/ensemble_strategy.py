import pandas as pd
import numpy as np
from abc import ABC


class EnsembleStrategy(ABC):
    def select_trial_ids(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile_threshold,
    ) -> list:
        return None


class RoundRobin(EnsembleStrategy):
    def select_trial_ids(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile_threshold,
    ) -> list:
        if configurations_to_select < 0:
            return [x for x in set(results) if x in results_df.index]
        selected_trial_ids = []
        groupby_list = [
            f"config.{k}" for k in pipeline_blueprint.get_all_distributions().keys()
        ]
        percentile = np.percentile(
            results_df["mean_validation_score"], percentile_threshold
        )
        groupby_list.reverse()
        grouped_results_df = results_df.sort_values(
            by="mean_validation_score", ascending=False
        ).groupby(by=groupby_list)
        group_dfs = [
            group.sort_values(by="mean_validation_score", ascending=False)
            for name, group in grouped_results_df
            if group["mean_validation_score"].max() >= percentile
        ]
        idx = 0
        iter = True
        while iter and any(len(group) > idx for group in group_dfs):
            for group in group_dfs:
                if len(selected_trial_ids) >= configurations_to_select:
                    iter = False
                    break
                if len(group) <= idx:
                    continue
                print(selected_trial_ids)
                print(idx)
                print(len(group))
                print(group.iloc[idx].name)
                selected_trial_ids.append(results[group.iloc[idx].name]["config"])
            idx += 1
        return selected_trial_ids


class RoundRobinEstimator(EnsembleStrategy):
    def select_trial_ids(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile_threshold,
    ) -> list:
        if configurations_to_select < 0:
            return [x for x in set(results) if x in results_df.index]
        selected_trial_ids = []
        groupby_list = ["config.Estimator"]
        percentile = np.percentile(
            results_df["mean_validation_score"], percentile_threshold
        )
        grouped_results_df = results_df.sort_values(
            by="mean_validation_score", ascending=False
        ).groupby(by=groupby_list)
        group_dfs = [
            group.sort_values(by="mean_validation_score", ascending=False)
            for name, group in grouped_results_df
        ]
        group_dfs.sort(key=lambda x: x["mean_validation_score"].max(), reverse=True)
        group_dfs = [group_df for group_df in group_dfs if group_df.iloc[0]["mean_validation_score"] >= percentile]
        idx = 0
        iter = True
        while iter and any(len(group) > idx for group in group_dfs):  # TODO optimize
            for group in group_dfs:
                print(f"round robin group len: {len(group)}")
                if len(selected_trial_ids) >= configurations_to_select:
                    iter = False
                    break
                if len(group) <= idx:
                    continue
                selected_trial_ids.append(group.iloc[idx].name)
            idx += 1
        print(f"round robin trial_ids: {selected_trial_ids}")
        return selected_trial_ids


class EnsembleBest(EnsembleStrategy):
    def select_trial_ids(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile_threshold,
    ) -> list:
        return None


class OneRoundRobinThenEnsembleBest(EnsembleStrategy):
    def select_trial_ids(
        self,
        results: dict,
        results_df: pd.DataFrame,
        configurations_to_select: int,
        pipeline_blueprint,
        percentile_threshold,
    ) -> list:
        return None
