import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
class Config:
    def __init__(self):
        # Paper-specific constants
        self.velocity_threshold = 0.5  # From velocity-threshold algorithm
        self.flow_theory_factor = 0.8  # From Flow Theory
        self.learning_rate = 0.1

        # Reward parameters
        self.base_reward = 10
        self.performance_weight = 0.6
        self.effort_weight = 0.3
        self.consistency_weight = 0.1

        # Other settings
        self.max_iterations = 1000


# Main class for reward system
class RewardSystem:
    def __init__(self, config: Config):
        self.config = config
        self.iterations = 0
        self.total_reward = 0
        self.rewards = []
        self.performance_scores = []
        self.effort_scores = []
        self.consistency_scores = []

    def calculate_reward(self, performance: float, effort: float, consistency: float) -> float:
        """
        Calculate the reward based on performance, effort, and consistency scores.
        Args:
            performance (float): Learner's performance score.
            effort (float): Learner's effort score.
            consistency (float): Learner's consistency score.
        Returns:
            float: Calculated reward.
        """
        # Validate input scores
        self._validate_scores(performance, effort, consistency)

        # Calculate weighted sum of scores
        weighted_performance = performance * self.config.performance_weight
        weighted_effort = effort * self.config.effort_weight
        weighted_consistency = consistency * self.config.consistency_weight
        weighted_sum = weighted_performance + weighted_effort + weighted_consistency

        # Apply Flow Theory factor
        adjusted_sum = weighted_sum * self.config.flow_theory_factor

        # Ensure reward is within acceptable range
        reward = np.clip(adjusted_sum, 0, self.config.base_reward)

        # Update total reward and reward history
        self.total_reward += reward
        self.rewards.append(reward)

        logger.info(f"Iteration {self.iterations}: Performance={performance}, Effort={effort}, Consistency={consistency}, Reward={reward}")

        return reward

    def velocity_threshold_algorithm(self, current_performance: float, previous_performance: float) -> float:
        """
        Apply the velocity-threshold algorithm to determine the effort score.
        Args:
            current_performance (float): Learner's current performance score.
            previous_performance (float): Learner's previous performance score.
        Returns:
            float: Effort score.
        """
        effort = 0
        if previous_performance is not None:
            velocity = (current_performance - previous_performance) / self.config.learning_rate
            if velocity > self.config.velocity_threshold:
                effort = 1

        return effort

    def update_performance(self, current_performance: float, previous_performance: float) -> None:
        """
        Update the performance score and history.
        Args:
            current_performance (float): Learner's current performance score.
            previous_performance (float): Learner's previous performance score.
        """
        self.performance_scores.append(current_performance)
        if previous_performance is not None:
            velocity = (current_performance - previous_performance) / self.config.learning_rate
            logger.debug(f"Performance velocity: {velocity}")

    def update_effort(self, current_effort: float) -> None:
        """
        Update the effort score and history.
        Args:
            current_effort (float): Learner's current effort score.
        """
        self.effort_scores.append(current_effort)

    def update_consistency(self, is_consistent: bool) -> None:
        """
        Update the consistency score and history.
        Consistency is represented as a binary value (0 or 1).
        Args:
            is_consistent (bool): True if learner's behavior is consistent, False otherwise.
        """
        consistency = 1 if is_consistent else 0
        self.consistency_scores.append(consistency)

    def _validate_scores(self, performance: float, effort: float, consistency: float) -> None:
        """
        Validate the input scores to ensure they are within acceptable ranges.
        Args:
            performance (float): Learner's performance score.
            effort (float): Learner's effort score.
            consistency (float): Learner's consistency score.
        Raises:
            ValueError: If any score is outside the valid range.
        """
        if not (0 <= performance <= 1):
            raise ValueError("Performance score must be between 0 and 1.")
        if not (0 <= effort <= 1):
            raise ValueError("Effort score must be between 0 and 1.")
        if not isinstance(consistency, int) or not (0 <= consistency <= 1):
            raise ValueError("Consistency score must be an integer between 0 and 1.")

    def reset(self) -> None:
        """
        Reset the reward system for a new episode/iteration.
        """
        self.iterations += 1
        self.total_reward = 0
        self.rewards = []
        self.performance_scores = []
        self.effort_scores = []
        self.consistency_scores = []
        logger.info(f"Starting new iteration: {self.iterations}")

# Helper class to manage data persistence
class DataPersistence:
    def __init__(self):
        self.data = pd.DataFrame()

    def save_rewards(self, rewards: List[float], iteration: int) -> None:
        """
        Save reward data for the current iteration.
        Args:
            rewards (List[float]): List of rewards for the current iteration.
            iteration (int): Current iteration number.
        """
        self.data[f"Iteration {iteration}"] = rewards

    def save_performance(self, performance_scores: List[float], iteration: int) -> None:
        """
        Save performance data for the current iteration.
        Args:
            performance_scores (List[float]): List of performance scores for the current iteration.
            iteration (int): Current iteration number.
        """
        self.data[f"Performance {iteration}"] = performance_scores

    def save_effort(self, effort_scores: List[float], iteration: int) -> None:
        """
        Save effort data for the current iteration.
        Args:
            effort_scores (List[float]): List of effort scores for the current iteration.
            iteration (int): Current iteration number.
        """
        self.data[f"Effort {iteration}"] = effort_scores

    def save_consistency(self, consistency_scores: List[int], iteration: int) -> None:
        """
        Save consistency data for the current iteration.
        Args:
            consistency_scores (List[int]): List of consistency scores for the current iteration.
            iteration (int): Current iteration number.
        """
        self.data[f"Consistency {iteration}"] = consistency_scores

    def export_data(self, filename: str) -> None:
        """
        Export the collected data to a CSV file.
        Args:
            filename (str): Name of the output file.
        """
        self.data.to_csv(filename, index=False)

# Exception classes
class InvalidScoreError(ValueError):
    pass

class MaxIterationsExceededError(RuntimeError):
    pass

# Integration interface functions
def calculate_rewards(performance_data: Dict[int, float], effort_data: Dict[int, float], consistency_data: Dict[int, int]) -> Dict[int, float]:
    """
    Calculate rewards for each iteration based on performance, effort, and consistency scores.
    Args:
        performance_data (Dict[int, float]): Mapping of iteration number to performance score.
        effort_data (Dict[int, float]): Mapping of iteration number to effort score.
        consistency_data (Dict[int, int]): Mapping of iteration number to consistency score.
    Returns:
        Dict[int, float]: Mapping of iteration number to calculated reward.
    """
    rewards = {}
    for iteration, (performance, effort, consistency) in enumerate(zip(performance_data.values(), effort_data.values(), consistency_data.values())):
        reward_system = RewardSystem(Config())
        reward = reward_system.calculate_reward(performance, effort, consistency)
        rewards[iteration] = reward
    return rewards

def main() -> None:
    # Initialize reward system and data persistence
    config = Config()
    reward_system = RewardSystem(config)
    data_persistence = DataPersistence()

    # Simulated data for demonstration
    performance_scores = [0.5, 0.6, 0.7, 0.8, 0.75]
    effort_scores = [0.4, 0.5, 0.6, 0.7, 0.65]
    consistency_scores = [1, 1, 0, 1, 1]

    # Iterate through each iteration/episode
    for iteration in range(len(performance_scores)):
        # Update performance, effort, and consistency scores
        previous_performance = performance_scores[iteration - 1] if iteration > 0 else None
        reward_system.update_performance(performance_scores[iteration], previous_performance)
        reward_system.update_effort(effort_scores[iteration])
        reward_system.update_consistency(consistency_scores[iteration])

        # Calculate and log reward
        reward = reward_system.calculate_reward(performance_scores[iteration], effort_scores[iteration], consistency_scores[iteration])

        # Save data for the current iteration
        data_persistence.save_rewards([reward], iteration)
        data_persistence.save_performance([performance_scores[iteration]], iteration)
        data_persistence.save_effort([effort_scores[iteration]], iteration)
        data_persistence.save_consistency([consistency_scores[iteration]], iteration)

        # Reset reward system for the next iteration
        if iteration < len(performance_scores) - 1:
            reward_system.reset()

    # Export data to a CSV file
    data_persistence.export_data("reward_data.csv")

if __name__ == "__main__":
    main()