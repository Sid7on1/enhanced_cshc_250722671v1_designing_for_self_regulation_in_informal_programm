import numpy as np
import pandas as pd
from typing import List


class EvaluationMetrics:
    """
    Class for evaluating agent performance using various metrics.
    """

    def __init__(self):
        self._metrics = {}

    def _validate_input_data(self, X: np.array, y: np.array):
        if len(X) != len(y):
            raise ValueError("X and y arrays must have same length.")

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X should be a 2D array and y should be a 1D array representing labels."
            )

        self._n_samples = len(X)

    def accuracy(self, X: List[int], y: List[int]) -> float:
        """
        Calculate accuracy metric.

        Parameters:
            X (List[int]): Predicted labels.
            y (List[int]): Ground truth labels.

        Returns:
            float: Accuracy value.
        """
        self._validate_input_data(np.array(X), np.array(y))

        correct = sum(X == y)
        return correct / self._n_samples

    def precision(self, X: List[int], y: List[int]) -> float:
        """
        Calculate precision metric.

        Parameters:
            X (List[int]): Predicted labels.
            y (List[int]): Ground truth labels.

        Returns:
            float: Precision value.
        """
        true_positives = sum(np.logical_and(X == 1, y == 1))
        false_positives = sum(np.logical_and(X == 1, y == 0))

        precision = true_positives / (true_positives + false_positives) if (true_positives > 0) else 0.0
        return precision

    def recall(self, X: List[int], y: List[int]) -> float:
        """
        Calculate recall metric.

        Parameters:
            X (List[int]): Predicted labels.
            y (List[int]): Ground truth labels.

        Returns:
            float: Recall value.
        """
        true_positives = sum(np.logical_and(X == 1, y == 1))
        false_negatives = sum(np.logical_and(X == 0, y == 1))

        recall = true_positives / (true_positives + false_negatives) if (true_positives > 0) else 0.0
        return recall

    def f1_score(self, X: List[int], y: List[int]) -> float:
        """
        Calculate F1 score metric.

        Parameters:
            X (List[int]): Predicted labels.
            y (List[int]): Ground truth labels.

        Returns:
            float: F1 score value.
        """
        precision = self.precision(X, y)
        recall = self.recall(X, y)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def flow_efficiency(self, X: List[float], Y: List[float], T: float = 1.5) -> float:
        """
        Calculate Flow Efficiency metric as per the research paper.

        Parameters:
            X (List[float]): List of task completion times.
            Y (List[float]): List of corresponding task switching times.
            T (float): Threshold value. Default is 1.5.

        Returns:
            float: Flow Efficiency value.
        """
        if len(X) != len(Y):
            raise ValueError("X and Y lists must have same length.")

        fe = 0
        for i in range(len(X)):
            if X[i] <= T:
                if Y[i] < X[i]:
                    fe += 1

        return fe / len(X)

    def velocity_threshold(self, X: List[float]) -> float:
        """
        Calculate Velocity Threshold metric as per the research paper.

        Parameters:
            X (List[float]): List of task completion times.

        Returns:
            float: Velocity Threshold value.
        """
        if not X:
            return 0.0

        X = np.array(X)
        mean_time = np.mean(X)
        std_dev = np.std(X)

        threshold = 2 * std_dev + mean_time
        return threshold

    def evaluate_agent(self, X: List[List[int]], Y: List[List[int]]) -> pd.DataFrame:
        """
        Evaluate the agent's performance using multiple metrics and return a DataFrame.

        Parameters:
            X (List[List[int]]): List of predicted actions/labels for each episode.
            Y (List[List[int]]): List of ground truth actions/labels for each episode.

        Returns:
            pd.DataFrame: DataFrame containing evaluation metrics.
        """
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        self._metrics = {metric: 0.0 for metric in metrics}

        for x, y in zip(X, Y):
            self.accuracy(x, y)
            self.precision(x, y)
            self.recall(x, y)
            self.f1_score(x, y)

        df = pd.DataFrame(self._metrics, index=[0])
        df.index.name = "Metric"

        return df

    def add_metric(self, metric_name: str, value: float):
        """
        Add a custom metric value to the evaluation results.

        Parameters:
            metric_name (str): Name of the metric.
            value (float): Metric value to be added.
        """
        self._metrics[metric_name] = value

    def get_metrics(self) -> dict:
        """
        Retrieve the computed metrics as a dictionary.

        Returns:
            dict: Metrics and their corresponding values.
        """
        return self._metrics


class AgentEvaluation:
    """
    Class for managing agent evaluation processes.
    """

    def __init__(self):
        self.eval_metrics = EvaluationMetrics()

    def perform_evaluation(
        self,
        X: List[List[int]],
        Y: List[List[int]],
        episode_data: List[dict],
        switch_costs: List[float],
        ep_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Perform comprehensive evaluation of the agent.

        Parameters:
            X (List[List[int]]): List of predicted actions/labels for each episode.
            Y (List[List[int]]): List of ground truth actions/labels for each episode.
            episode_data (List[dict]): Additional episode data for metrics calculation.
            switch_costs (List[float]): List of switch costs corresponding to episodes.
            ep_threshold (float): Episode threshold value for Flow Efficiency metric. Default is 1.5.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        total_cost = 0

        df_metrics = self.eval_metrics.evaluate_agent(X, Y)

        for i, (ep_x, ep_y) in enumerate(zip(X, Y)):
            ep_data = episode_data[i]
            switch_cost = switch_costs[i]

            # Calculate Flow Efficiency
            task_times = ep_data["task_times"]
            switching_times = ep_data["switching_times"]
            fe_value = self.eval_metrics.flow_efficiency(task_times, switching_times, ep_threshold)
            df_metrics.loc["Flow Efficiency"] = fe_value

            # Calculate Velocity Threshold
            threshold = self.eval_metrics.velocity_threshold(task_times)
            df_metrics.loc["Velocity Threshold"] = threshold

            # Calculate total cost for the episode
            total_cost += switch_cost

            self.eval_metrics.add_metric("Episode Cost", switch_cost)

            # Add episode number as a metric for convenience
            df_metrics.loc["Episode"] = i + 1

        df_metrics.loc["Total Cost"] = total_cost

        return df_metrics


AgentEvaluationInstance = AgentEvaluation()

# Example usage:
X_pred = [
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
]  # Predicted labels for 5 episodes

Y_true = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1],
]  # Ground truth labels

EpisodeData = [
    {"task_times": [2.5, 1.7, 3.2], "switching_times": [1.2, 2.1, 2.9]},
    {"task_times": [1.8, 1.2, 1.5], "switching_times": [1.1, 1.3, 1.4]},
    {"task_times": [2.9, 2.3, 2.7], "switching_times": [1.6, 2.2, 2.6]},
    {"task_times": [1.4, 1.9, 1.2], "switching_times": [1.3, 1.7, 1.1]},
    {"task_times": [2.1, 1.6, 2.4], "switching_times": [1.4, 1.9, 2.3]},
]  # Additional episode data

SwitchCosts = [0.8, 0.6, 0.9, 0.7, 0.85]  # Switch costs for each episode

results_df = AgentEvaluationInstance.perform_evaluation(
    X_pred, Y_true, EpisodeData, SwitchCosts, ep_threshold=1.6
)

print(results_df)
"""

This code implements the required evaluation metrics and manages the evaluation process for the agent-based project. It integrates the specifications mentioned in the research paper, incorporating Flow Efficiency and Velocity Threshold metrics. The code is structured with appropriate classes, methods, and production-ready considerations.