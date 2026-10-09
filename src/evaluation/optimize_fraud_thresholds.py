import numpy as np
import json
from pathlib import Path


def optimize_fraud_thresholds(
    y_prob: np.ndarray,
    amounts: np.ndarray,
    y_true: np.ndarray = None,
    cb_fee: float = 15.0,
    review_cost: float = 2.50,
    churn_cost: float = 10.0,
    catch_human: float = 0.90,
    max_review_rate: float = 0.085,    # Hard business constraint: max 8.5% to review
    min_approve_rate: float = 0.85,   # Hard business constraint: min 85% auto-approved
    t_app_steps: int = 30,
    t_dec_steps: int = 30,
    save_path: str = "artifacts/thresholds.json"
) -> dict:
    """
    Finds the (t_approve, t_decline) pair that minimizes total financial portfolio loss
    using vectorized Bayes expected-cost formulations and breakeven gating.
    """
    p = np.asarray(y_prob, dtype=np.float64)
    amt = np.asarray(amounts, dtype=np.float64)

    # 1. Precalculate Vectorized Expected Cost Elements
    # E[Cost_Approve] = p * (Amt + Fee_cb)
    e_cost_app = p * (amt + cb_fee)
    
    # E[Cost_Decline] = (1 - p) * Cost_churn
    e_cost_dec = (1.0 - p) * churn_cost
    
    # E[Cost_Review] = C_audit + p * (1 - CatchRate_human) * (Amt + Fee_cb)
    e_cost_rev = review_cost + p * (1.0 - catch_human) * (amt + cb_fee)

    # 2. Vectorized Breakeven Gating Precomputation
    # When (Amt + Fee_cb) <= review_cost, human inspection is guaranteed negative ROI
    cannot_review = (amt + cb_fee) <= review_cost
    prefer_app_over_dec = e_cost_app <= e_cost_dec

    # 3. Grid Definition
    # t_approve  in low probability ranges 
    t_app_grid = np.linspace(0.005, 0.2, t_app_steps)
    # t_decline spans from moderate to high probability ranges 
    t_dec_grid = np.linspace(0.2, 0.45, t_dec_steps)

    best_loss = float("inf")
    best_t_app = None
    best_t_dec = None
    best_breakdown = {}

    total_n = len(p)

    # 4. Vectorized 2D Grid Evaluation
    for t_app in t_app_grid:
        is_approve_base = p < t_app

        for t_decline in t_dec_grid:
            if t_decline <= t_app:
                continue

            is_decline_base = p >= t_decline
            is_review_base = (~is_approve_base) & (~is_decline_base)

            # Apply Breakeven Review Bypass
            rev_bypass = is_review_base & cannot_review
            is_review = is_review_base & (~cannot_review)
            is_approve = is_approve_base | (rev_bypass & prefer_app_over_dec)
            is_decline = is_decline_base | (rev_bypass & (~prefer_app_over_dec))

            # Operational capacity check
            review_rate = np.sum(is_review) / total_n
            approve_rate = np.sum(is_approve) / total_n

            # Reject thresholds that overwhelm staff or customer checkout flow
            if review_rate > max_review_rate or approve_rate < min_approve_rate:
                continue

            # Financial realization
            loss_approved = np.sum(e_cost_app[is_approve])
            loss_reviewed = np.sum(e_cost_rev[is_review])
            loss_declined = np.sum(e_cost_dec[is_decline])
            
            total_loss = loss_approved + loss_reviewed + loss_declined

            if total_loss < best_loss:
                best_loss = total_loss
                best_t_app = float(t_app)
                best_t_dec = float(t_decline)
                
                # Capture operational snapshot
                n_app = int(np.sum(is_approve))
                n_rev = int(np.sum(is_review))
                n_dec = int(np.sum(is_decline))

                best_breakdown = {
                    "t_approve": best_t_app,
                    "t_decline": best_t_dec,
                    # "minimized_total_cost": float(total_loss),
                    # "auto_approve_rate": float(n_app / total_n),
                    # "manual_review_rate": float(n_rev / total_n),
                    # "hard_decline_rate": float(n_dec / total_n),
                    # "manual_review_cost": float(n_rev * review_cost),
                    # "prevented_fraud_exposure": float(np.sum(p[is_decline] * (amt[is_decline] + cb_fee))),
                    # "expected_fraud_loss": float(loss_approved + np.sum(p[is_review] * (1.0 - catch_human) * (amt[is_review] + cb_fee))),
                    # "churn_cost_loss": float(loss_declined)
                }

    # 5. Export optimal policy to artifacts
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds": {
                "t_approve": best_t_app,
                "t_decline": best_t_dec
            },
            "economics": {
                "chargeback_fee": cb_fee,
                "manual_review_cost": review_cost,
                "churn_cost": churn_cost,
                "human_catch_rate": catch_human
            },
            #"validation_performance": best_breakdown
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Optimal policy saved to {out_file}")

    return best_breakdown