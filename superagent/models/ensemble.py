"""Seed-averaged final model.

The search stage scores candidates with a single training seed. The deployed
model is then refitted on train+validation, and a single refit inherits the
seed's variance: on the headline-pair split two seeds of the same policy
differed by about one accuracy point on a 3232-row held-out set. This wrapper
keeps several refits of the *same* candidate and averages their predicted
probabilities, which lowers that variance without changing the candidate the
planner chose.

Only the final fit is repeated. Trials stay single-seed, so the search cost is
unchanged.
"""

from __future__ import annotations

import numpy as np


class SeedAveragedModel:
    """Average the predicted probabilities of fitted pipelines sharing a candidate."""

    def __init__(self, members, seeds):
        if not members:
            raise ValueError("SeedAveragedModel needs at least one fitted member")
        self.members = list(members)
        self.seeds = list(seeds)
        reference = [str(c) for c in self.members[0].classes_]
        for member in self.members[1:]:
            if [str(c) for c in member.classes_] != reference:
                raise ValueError("Seed members disagree on the class order")
        self.classes_ = self.members[0].classes_

    @property
    def named_steps(self):
        """Expose the first member's steps so model introspection keeps working."""
        return self.members[0].named_steps

    def __len__(self):
        return len(self.members)

    def predict_proba(self, frame):
        total = None
        for member in self.members:
            probability = np.asarray(member.predict_proba(frame), dtype=float)
            total = probability if total is None else total + probability
        return total / len(self.members)

    def predict(self, frame):
        probability = self.predict_proba(frame)
        return np.asarray(self.classes_)[np.argmax(probability, axis=1)]

    def member_scores(self, frame, target):
        """Report per-seed accuracy so the artifact records its own spread."""
        from sklearn.metrics import accuracy_score

        truth = np.asarray(target)
        return [float(accuracy_score(member.predict(frame), truth)) for member in self.members]

    def __repr__(self):
        return f"SeedAveragedModel(members={len(self.members)}, seeds={self.seeds})"
