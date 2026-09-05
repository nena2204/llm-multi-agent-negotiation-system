from typing import Dict, Any


class Judge:
    """Legacy price-only judge retained for CLI compatibility.

    New research evaluation belongs in :mod:`llm_negotiation.evaluation`.
    """

    def evaluate(self, history, deal_reached: bool, final_price, buyer, seller, rounds_used: int) -> Dict[str, Any]:
        result = {
            "deal_reached": deal_reached,
            "final_price": final_price,
            "rounds_used": rounds_used,
            "is_valid_price": None,
            "fairness_score": None,
            "explanation": "",
        }

        if deal_reached and final_price is not None:
            # valid price if in between seller min and buyer max
            is_valid = seller.min_acceptable <= final_price <= buyer.max_price
            result["is_valid_price"] = is_valid

            # fairness: distance from midpoint between buyer and seller bounds
            low = seller.min_acceptable
            high = buyer.max_price
            if high <= low:
                midpoint = (low + high) / 2.0
            else:
                midpoint = (low + high) / 2.0
            # normalized closeness to midpoint
            if high - low == 0:
                fairness = 100.0
            else:
                fairness = max(0.0, 100.0 - (abs(final_price - midpoint) / (high - low) * 100.0))
            result["fairness_score"] = round(fairness, 2)

            explanation = f"Deal reached at ${final_price:.2f} after {rounds_used} rounds."
            explanation += " Price is valid." if is_valid else " Price is outside acceptable bounds."
            result["explanation"] = explanation
        else:
            result["is_valid_price"] = False
            result["fairness_score"] = 0.0
            result["explanation"] = "No deal was reached."

        return result

