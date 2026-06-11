from typing import Dict, Any


class LearningAgent:
    """Simple reward evaluator for negotiation outcomes."""

    def score_result(self, evaluation: Dict[str, Any]) -> int:
        score = 0
        if evaluation.get("deal_reached"):
            score += 10
            # fair if fairness_score >= 60
            if evaluation.get("fairness_score", 0) >= 60:
                score += 5
            # if reached quickly (<=3 rounds), reward small bonus
            if evaluation.get("rounds_used", 999) <= 3:
                score += 3
        else:
            score -= 5

        # penalty if fairness low when deal reached
        if evaluation.get("deal_reached") and evaluation.get("fairness_score", 0) < 40:
            score -= 10

        return score

