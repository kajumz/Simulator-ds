from typing import List, Tuple
import hashlib
import random
import bisect
from itertools import accumulate



class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
            self,
            experiment_id: int,
            groups: Tuple[str] = ("A", "B"),
            group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = hashlib.sha256(str(experiment_id).encode()).hexdigest()

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.
        if group_weights is None:
            self.group_weights = [1 / len(groups) for _ in groups]
        else:
            if len(group_weights) != len(groups) or not all(w >= 0 for w in group_weights):
                raise ValueError("Invalid group weights.")
            if sum(group_weights) != 1:
                raise ValueError("Group weights must sum to 1.")
            self.group_weights = group_weights

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name
        random.seed(self.salt + str(click_id))
        group_id = random.choices(range(len(self.groups)), weights=self.group_weights)[0]

        return group_id, self.groups[group_id]