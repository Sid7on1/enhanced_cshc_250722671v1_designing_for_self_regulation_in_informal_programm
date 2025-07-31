import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    from agent.algorithms import VelocityThreshold, FlowTheory
    from agent.utils import Experience, PrioritizedReplayBuffer, make_transition
    from agent.configuration import Config
    from agent.models import XRModel

except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    raise

class Memory:
    """
    Experience replay and memory module for the XR agent.
    Handles experience storage, retrieval, and sampling for training.
    """

    def __init__(self, capacity: int, batch_size: int, priority_fraction: float = 0.0,
                 priority_sampling: bool = False, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, abs_error_upper: float = 1.0,
                 algorithm: str = "velocity-threshold", *args, **kwargs):
        """
        Initializes the Memory module.

        :param capacity: Capacity of the experience replay buffer.
        :param batch_size: Batch size for sampling experiences.
        :param priority_fraction: Fraction of experiences to be sampled prioritely.
        :param priority_sampling: Whether to use priority sampling or not.
        :param alpha: Priority exponent for prioritized experience replay.
        :param beta: Initial importance sampling weight for prioritized replay.
        :param beta_increment: Increment factor for beta.
        :param abs_error_upper: Upper bound for absolute error in prioritized replay.
        :param algorithm: Algorithm to use for experience replay, choices: ["velocity-threshold", "flow-theory"].
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.priority_fraction = priority_fraction
        self.priority_sampling = priority_sampling
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.abs_error_upper = abs_error_upper
        self.algorithm = algorithm

        # Initialize experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity, alpha)

        # Initialize algorithm-specific parameters
        self._init_algorithm_params()

        # XR-specific parameters
        self.state_size = Config.state_size
        self.action_size = Config.action_size
        self.device = Config.device

        # Data structures for XR-specific experience storage
        self.eye_tracking_data = []
        self.contextual_data = []
        self.user_interactions = []

        # Models for algorithm-specific computations
        self.velocity_threshold_model = VelocityThreshold()
        self.flow_theory_model = FlowTheory()

    def _init_algorithm_params(self):
        """
        Initializes parameters specific to the chosen algorithm.
        """
        if self.algorithm == "velocity-threshold":
            # Velocity threshold algorithm parameters
            self.threshold = 0.5  # Example threshold value
            self.velocity_window = 5  # Number of consecutive experiences to consider

        elif self.algorithm == "flow-theory":
            # Flow theory algorithm parameters
            self.challenge_threshold = 0.7  # Example challenge threshold
            self.skill_threshold = 0.6  # Example skill threshold
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool, **kwargs) -> None:
        """
        Stores a single experience in the replay buffer.

        :param state: Current state observation.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state observation.
        :param done: Whether the episode is done or not.
        :param kwargs: Additional keyword arguments for XR-specific data.
        """
        # XR-specific data extraction
        eye_tracking_data = kwargs.get("eye_tracking_data")
        contextual_data = kwargs.get("contextual_data")
        user_interactions = kwargs.get("user_interactions")

        # Store XR-specific data separately for later analysis
        self.eye_tracking_data.append(eye_tracking_data)
        self.contextual_data.append(contextual_data)
        self.user_interactions.append(user_interactions)

        # Convert state and action to torch tensors
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.tensor([action]).long().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.tensor([done]).bool().to(self.device)
        reward = torch.tensor([reward]).float().to(self.device)

        # Create an Experience object
        transition = make_transition(state, action, next_state, reward, done)

        # Store experience in the replay buffer
        self.replay_buffer.add(transition)

    def sample_experiences(self) -> DataLoader:
        """
        Samples a batch of experiences from the replay buffer.

        :return: DataLoader object containing the sampled experiences.
        """
        if self.priority_sampling:
            # Sample experiences based on priorities
            batch, indices, priorities = self.replay_buffer.sample(self.batch_size, self.beta)
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Sample experiences uniformly
            batch = self.replay_buffer.sample(self.batch_size)
            indices = None
            priorities = None

        # Create a DataLoader for the sampled experiences
        dataset = Experience(batch)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return data_loader, indices, priorities

    def update_experience_priorities(self, indices: List[int], errors: np.ndarray) -> None:
        """
        Updates the priorities of experiences based on their TD errors.

        :param indices: List of indices of the sampled experiences.
        :param errors: TD errors corresponding to the sampled experiences.
        """
        # Clip errors to prevent large outliers from dominating the prioritization
        errors = np.clip(errors, -self.abs_error_upper, self.abs_error_upper)

        # Update priorities of the sampled experiences
        for idx, error in zip(indices, errors):
            self.replay_buffer.update(idx, error**self.alpha)

    def compute_engagement_score(self, experiences: List[Experience]) -> np.ndarray:
        """
        Computes the engagement score for a list of experiences using the chosen algorithm.

        :param experiences: List of Experience objects.
        :return: Numpy array of engagement scores corresponding to each experience.
        """
        # Extract states and rewards from experiences
        states = [exp.state for exp in experiences]
        rewards = [exp.reward for exp in experiences]

        # Convert to numpy arrays
        states = np.array(states)
        rewards = np.array(rewards)

        if self.algorithm == "velocity-threshold":
            # Compute engagement score using the velocity threshold algorithm
            engagement_scores = self._compute_engagement_velocity_threshold(states, rewards)

        elif self.algorithm == "flow-theory":
            # Compute engagement score using the flow theory algorithm
            engagement_scores = self._compute_engagement_flow_theory(states, rewards)

        return engagement_scores

    def _compute_engagement_velocity_threshold(self, states: np.ndarray,
                                             rewards: np.ndarray) -> np.ndarray:
        """
        Computes the engagement score using the velocity threshold algorithm.

        :param states: Numpy array of states.
        :param rewards: Numpy array of rewards.
        :return: Numpy array of engagement scores.
        """
        # Compute velocity of rewards
        reward_velocity = np.diff(rewards)

        # Smoothen the velocity using a moving average
        smoothed_velocity = np.convolve(reward_velocity, np.ones(self.velocity_window), mode='valid') / self.velocity_window

        # Compute engagement score based on the velocity threshold
        engagement_scores = np.where(smoothed_velocity > self.threshold, 1, 0)

        return engagement_scores

    def _compute_engagement_flow_theory(self, states: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Computes the engagement score using the flow theory algorithm.

        :param states: Numpy array of states.
        :param rewards: Numpy array of rewards.
        :return: Numpy array of engagement scores.
        """
        # Compute challenge and skill levels based on states and rewards
        challenge_levels = self.flow_theory_model.compute_challenge_level(states)
        skill_levels = self.flow_theory_model.compute_skill_level(rewards)

        # Compute engagement score based on challenge and skill thresholds
        engagement_scores = np.where((challenge_levels > self.challenge_threshold) &
                                    (skill_levels > self.skill_threshold), 1, 0)

        return engagement_scores

    def analyze_xr_data(self) -> Dict[str, pd.DataFrame]:
        """
        Analyzes the separately stored XR-specific data and returns insights.

        :return: Dictionary containing XR-specific data insights.
        """
        # TODO: Implement XR-specific data analysis here
        # For example, you can analyze eye tracking data to compute fixation durations,
        # or analyze user interactions to identify patterns in user behavior.
        # Return relevant insights as pandas dataframes or dictionaries.

        # Placeholder implementation
        eye_tracking_insights = pd.DataFrame()  # Replace with actual insights
        contextual_insights = pd.DataFrame()  # Replace with actual insights
        user_behavior_insights = pd.DataFrame()  # Replace with actual insights

        return {
            "eye_tracking_data": eye_tracking_insights,
            "contextual_data": contextual_insights,
            "user_interactions": user_behavior_insights
        }

    def get_memory_state(self) -> Dict[str, object]:
        """
        Returns the current state of the memory module.

        :return: Dictionary containing the state of the memory module.
        """
        state = {
            "replay_buffer": self.replay_buffer.get_state(),
            "algorithm": self.algorithm,
            "eye_tracking_data": self.eye_tracking_data,
            "contextual_data": self.contextual_data,
            "user_interactions": self.user_interactions
        }
        return state

    def set_memory_state(self, state: Dict[str, object]) -> None:
        """
        Sets the state of the memory module.

        :param state: Dictionary containing the state of the memory module.
        """
        self.replay_buffer.set_state(state["replay_buffer"])
        self.algorithm = state["algorithm"]
        self.eye_tracking_data = state["eye_tracking_data"]
        self.contextual_data = state["contextual_data"]
        self.user_interactions = state["user_interactions"]

    def get_engagement_metrics(self) -> Dict[str, float]:
        """
        Computes and returns engagement metrics based on the stored experiences.

        :return: Dictionary containing engagement metrics.
        """
        # Compute engagement scores for all experiences
        experiences = self.replay_buffer.get_all_transitions()
        engagement_scores = self.compute_engagement_score(experiences)

        # Compute engagement metrics
        num_engaged_experiences = np.sum(engagement_scores)
        engagement_ratio = num_engaged_experiences / len(engagement_scores)

        return {
            "num_engaged_experiences": num_engaged_experiences,
            "engagement_ratio": engagement_ratio
        }


if __name__ == "__main__":
    # Example usage
    memory = Memory(capacity=1000, batch_size=32, algorithm="velocity-threshold")

    # Storing experiences
    state = np.array([0.1, 0.2, 0.3])
    action = 0
    reward = 1.0
    next_state = np.array([0.4, 0.5, 0.6])
    done = False
    eye_tracking_data = {"fixation_duration": 2.5, "saccade_amplitude": 3.0}  # Example data
    contextual_data = {"user_id": "user123", "program_id": "prog456"}  # Example data
    user_interactions = {"click_count": 5, "scroll_distance": 200}  # Example data
    memory.store_experience(state, action, reward, next_state, done,
                           eye_tracking_data=eye_tracking_data,
                           contextual_data=contextual_data,
                           user_interactions=user_interactions)

    # Sampling experiences
    experiences, indices, priorities = memory.sample_experiences()
    for exp in experiences:
        print(exp)

    # Updating experience priorities
    td_errors = np.random.random(len(experiences))  # Example TD errors
    memory.update_experience_priorities(indices, td_errors)

    # Analyzing XR-specific data
    xr_insights = memory.analyze_xr_data()
    print("XR Insights:")
    print(xr_insights)

    # Getting and setting memory state
    memory_state = memory.get_memory_state()
    print("Memory State:")
    print(memory_state)

    new_memory = Memory(capacity=500, batch_size=64, algorithm="flow-theory")
    new_memory.set_memory_state(memory_state)

    # Getting engagement metrics
    engagement_metrics = memory.get_engagement_metrics()
    print("Engagement Metrics:")
    print(engagement_metrics)