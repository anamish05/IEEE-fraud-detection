import os
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from .duckdb_store import DuckDBStateStore
import dagshub
import mlflow
import mlflow.lightgbm
from src.config import DAGSHUB_USERNAME, DAGSHUB_REPO, artifacts_path
from dotenv import load_dotenv

class TransactionStreamer:

    def __init__(
        self,
        model_uri: str,
        features_path: str,
        target_enc_path: str,
        freq_enc_path: str,
        train_warmup_path: str,
        cat_features_path:str,
        thresholds_path: str=None
    ):
        # Initialize DB and warm up state
        self.store = DuckDBStateStore()
        print("Hydrating 24h training table...")
        self.store.warm_up(train_warmup_path)

        load_dotenv()

        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO,
            mlflow=True
        )

        mlflow.set_experiment("ieee-fraud-inference")
        print(f"Connected to MLflow: {mlflow.get_tracking_uri()}")

        # Load model and feature list
        print(f"Fetching model from: {model_uri}")
        self.model = mlflow.lightgbm.load_model(model_uri)

        with open(features_path, "r") as f:
            self.model_features = json.load(f)

        # Load frozen training encodings
        with open(target_enc_path, "r") as f:
            self.te_data = json.load(f)

        with open(freq_enc_path, "r") as f:
            self.freq_data = json.load(f)

        with open(os.path.join(artifacts_path, "v_clusters.json"), "r") as f:
            self.v_clusters=json.load(f)

        with open(cat_features_path, "r") as f:
            self.cat_features = json.load(f)

        if thresholds_path:
            with open(thresholds_path, 'r') as f:
                data = json.load(f)
            if "thresholds" in data:
                self.policy = {**data, **data['thresholds']}
            else:
                self.policy=data
        else:
            self.policy={'t_approve':0.05, 't_decline':0.35, "economics": {"chargeback_fee": 15.0,"manual_review_cost": 2.50,"churn_cost": 10.0,"human_catch_rate": 0.90}}

    def run_inference(self, test_trans_path: str, test_ident_path: str, output_csv: str, chunk_size: int = 50000):

        with mlflow.start_run(run_name="streaming-inference"):
            mlflow.log_params({
                "chunk_size": chunk_size,
                "feature_count": len(self.model_features)
            })


            # Create stream view
            self.store.con.execute(f"""
                CREATE OR REPLACE VIEW test_stream_view AS 
                SELECT 
                    t.*, 
                    i.id_01, i.id_02, i.id_03, i.id_04, i.id_05, i.id_06, i.id_07, i.id_08,
                    i.id_09, i.id_10, i.id_11, i.id_12, i.id_13, i.id_14, i.id_15, i.id_16,
                    i.id_17, i.id_18, i.id_19, i.id_20, i.id_21, i.id_22, i.id_23, i.id_24,
                    i.id_25, i.id_26, i.id_27, i.id_28, i.id_29, i.id_30, i.id_31, i.id_32,
                    i.id_33, i.id_34, i.id_35, i.id_36, i.id_37, i.id_38, i.DeviceType, i.DeviceInfo,
                    make_pseudo_uid(t.card1, t.card2, t.card3, t.card4, t.card5, t.card6, t.addr1, t.addr2, t.TransactionDT, t.D1) AS Pseudo_UID,
                    ('2017-12-01'::TIMESTAMP + to_seconds(t.TransactionDT::BIGINT)) AS event_time,
                    ((t.TransactionDT // 86400) - t.D1) AS card_birthday,
                    (t.TransactionDT // 3600) % 24 AS hour_of_day,
                    (t.TransactionDT // (3600 * 24)) % 7 AS day_of_week
                FROM read_csv('{test_trans_path}') t
                LEFT JOIN read_csv('{test_ident_path}') i ON t.TransactionID = i.TransactionID
                ORDER BY t.TransactionDT ASC;
            """)

            cursor = self.store.con.cursor()
            cursor.execute("SELECT * FROM test_stream_view;")

            predictions, tx_ids, actions = [], [], []
            transaction_amounts= []

            # Extract thresholds safely (works whether nested or flat)
            t_approve = (
                self.policy.get("thresholds", {}).get("t_approve")
                or self.policy.get("t_approve", 0.05)
            )
            t_decline = (
                self.policy.get("thresholds", {}).get("t_decline")
                or self.policy.get("t_decline", 0.35)
            )

            while True:
                chunk_df = cursor.fetch_df_chunk(chunk_size)
                if chunk_df.empty:
                    break

                # Register chunk in DuckDB for zero-copy SQL access
                self.store.con.register("current_chunk", chunk_df)

                chunk_with_features_df = self.store.con.execute("""
                    WITH combined_stream AS (
                        -- Active historical state
                        SELECT 
                            -1::BIGINT AS TransactionID, Pseudo_UID, event_time, amount as TransactionAmt, 
                            FALSE AS is_chunk
                        FROM active_user_store
                        
                        UNION ALL
                        
                        -- Incoming chunk records
                        SELECT 
                            TransactionID, Pseudo_UID, event_time, TransactionAmt, 
                            TRUE AS is_chunk
                        FROM current_chunk
                    ),
                    rolling_stats AS (
                        SELECT 
                            TransactionID,
                            is_chunk,
                            -- Transactions in preceding 24 hours (excluding current record)
                            COUNT(TransactionAmt) OVER (
                                PARTITION BY Pseudo_UID 
                                ORDER BY event_time
                                RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
                            ) AS uid_count_24h,
                            
                            -- Mean amount in preceding 24 hours
                            COALESCE(
                                AVG(TransactionAmt) OVER (
                                    PARTITION BY Pseudo_UID 
                                    ORDER BY event_time
                                    RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
                                ), 0.0
                            ) AS uid_amt_mean_24h,
                            
                            -- Previous event within the current stream window
                            LAG(event_time) OVER (
                                PARTITION BY Pseudo_UID  
                                ORDER BY event_time, is_chunk, TransactionID
                            ) AS prev_event_time
                        FROM combined_stream
                    )
                    SELECT 
                        c.*,
                        s.uid_count_24h,
                        s.uid_amt_mean_24h,
                        COALESCE(
                            date_diff('second', s.prev_event_time, c.event_time),
                            date_diff('second', ult.last_event_time, c.event_time)
                        ) AS uid_dt_diff
                    FROM current_chunk c
                    JOIN rolling_stats s ON c.TransactionID = s.TransactionID
                    LEFT JOIN user_last_tx ult ON c.Pseudo_UID=ult.pseudo_uid
                    WHERE s.is_chunk = TRUE
                    ORDER BY c.TransactionDT ASC, c.TransactionID ASC;
                """).df()



                # Encodings
                for col, mapping in self.te_data["mappings"].items():
                    chunk_with_features_df[f"{col}_target_enc"] = chunk_with_features_df[col].astype(str).map(mapping).fillna(self.te_data["global_prior"])

                for col, mapping in self.freq_data.items():
                    if col in chunk_with_features_df.columns:
                        chunk_with_features_df[f"{col}_freq"] = chunk_with_features_df[col].astype(str).map(mapping).fillna(0.0)

                for block_name, v_cols in self.v_clusters.items():
                    chunk_with_features_df[block_name]=chunk_with_features_df[v_cols].notna().any(axis=1)

                chunk_with_features_df["TransactionDTDays"] = chunk_with_features_df["TransactionDT"] / 86400   


                # Predict
                X_chunk = chunk_with_features_df[self.model_features].copy()
              
                for col in self.cat_features:
                    if col not in X_chunk.columns:
                        raise ValueError(f"Categorical feature {col} is missing from inference data")
                    X_chunk[col] = X_chunk[col].astype("category")

                preds = self.model.predict(X_chunk)

                conditions = [preds < t_approve, (preds >= t_approve)&(preds < t_decline), preds >= t_decline] 

                choices=['APPROVE', 'REVIEW', 'DECLINE']
                batch_actions=np.select(conditions, choices, default='REVIEW')

                predictions.extend(preds)
                tx_ids.extend(chunk_with_features_df["TransactionID"])
                actions.extend(batch_actions)
                transaction_amounts.extend(chunk_with_features_df['TransactionAmt'])

                # Append batch into active_user_store
                self.store.con.execute("""
                    INSERT INTO active_user_store (pseudo_uid, event_time, amount)
                    SELECT Pseudo_UID, event_time, TransactionAmt
                    FROM current_chunk;
                """)

                # Update user_last_tx with latest timestamp per user in chunk
                self.store.con.execute("""
                    INSERT OR REPLACE INTO user_last_tx
                    SELECT 
                        Pseudo_UID AS pseudo_uid,
                        MAX(event_time) AS last_event_time
                    FROM current_chunk
                    GROUP BY 1;
                """)

                # Evict state
                self.store.prune_old_events(chunk_with_features_df["event_time"].max())

                self.store.con.unregister("current_chunk")

            sub_df = pd.DataFrame({"TransactionID": tx_ids, "isFraud": predictions, 'decision': actions, 'TransactionAmt': transaction_amounts})
            sub_df.to_csv(output_csv, index=False)
            print(f"Inference complete: output saved to {output_csv}")

            # BUSINESS IMPACT
            # Economic parameters from policy (with fallbacks)
            econ = self.policy.get("economics", {})
            cb_fee = econ.get("chargeback_fee", 15.0)
            review_cost = econ.get("manual_review_cost", 2.50)
            churn_cost = econ.get("churn_cost", 10.0)
            catch_human = econ.get("human_catch_rate", 0.90)

            p = np.asarray(predictions)
            amt = np.asarray(transaction_amounts)

            # cost business rules (vectorized)
            e_cost_app = p * (amt + cb_fee)
            e_cost_dec = (1.0 - p) * churn_cost


            is_approve = p < t_approve
            is_decline = p >= t_decline
            is_review = (~is_approve) & (~is_decline)

            # breakeven rule for manual review
            cannot_review = (amt+cb_fee) <= review_cost
            prefer_app_over_dec = e_cost_app<=e_cost_dec

            review_bypass = is_review & cannot_review  # for review, but econoically too cheap to review
            is_review = is_review & (~cannot_review)   # for review, excluding bypass transactions
            is_approve = is_approve | (review_bypass & prefer_app_over_dec)   # if approve is cheaper than decline for bypass transaction - approve it
            is_decline = is_decline | (review_bypass & (~prefer_app_over_dec))  # else - decline it

            # Assign Decision Labels
            actions = np.empty(len(p), dtype=object)
            actions[is_approve] = "APPROVE"
            actions[is_review] = "REVIEW"
            actions[is_decline] = "DECLINE"

            # Save Submissions CSV
            sub_df = pd.DataFrame({
                "TransactionID": tx_ids,
                "isFraud": p,
                "decision": actions,
                "TransactionAmt": amt
            })
            sub_df.to_csv(output_csv, index=False)
            print(f"Inference complete: output saved to {output_csv}")


            # Financial & Operational Metrics
            total_tx = len(sub_df)
            n_approved = int(np.sum(is_approve))
            n_reviewed = int(np.sum(is_review))
            n_declined = int(np.sum(is_decline))

            # Manual review queue operational expenditure
            total_review_expense = n_reviewed * review_cost

            # Capital prevented by automated decline
            prevented_fraud_exposure = float(np.sum(p[is_decline] * (amt[is_decline] + cb_fee)))

            # Residual expected fraud loss (approved transactions + missed by reviewers)
            approved_fraud_loss = float(np.sum(e_cost_app[is_approve]))
            reviewed_fraud_leakage = float(np.sum(p[is_review] * (1.0 - catch_human) * (amt[is_review] + cb_fee)))
            total_expected_fraud_loss = approved_fraud_loss + reviewed_fraud_leakage

            # Customer lifetime value lost from false declines (insult cost)
            total_churn_loss = float(np.sum(e_cost_dec[is_decline]))

            # Total economic portfolio loss under this policy
            total_portfolio_cost = total_expected_fraud_loss + total_review_expense + total_churn_loss

            summary_metrics = {
                "total_transactions": total_tx,
                "auto_approve_rate": n_approved / total_tx,
                "manual_review_rate": n_reviewed / total_tx,
                "hard_decline_rate": n_declined / total_tx,
                "total_euros_declined": float(np.sum(amt[is_decline])),
                "total_euros_routed_to_review": float(np.sum(amt[is_review])),
                "manual_review_operational_cost": float(total_review_expense),
                "prevented_fraud_loss": float(prevented_fraud_exposure),
                "expected_fraud_loss": float(total_expected_fraud_loss),
                "churn_friction_loss": float(total_churn_loss),
                "total_portfolio_cost": float(total_portfolio_cost),
                "policy_roi_per_review_euro": (prevented_fraud_exposure / total_review_expense) if total_review_expense > 0 else 0.0
            }

            print("\n=== BUSINESS DECISION SUMMARY ===")
            for k, v in summary_metrics.items():
                print(f"{k}: {v:,.4f}" if isinstance(v, float) else f"{k}: {v}")

            print("Uploading submission to DagsHub MLflow Artifacts...")
            mlflow.log_artifact(output_csv, artifact_path="predictions")
            mlflow.log_metrics({
                "total_scored_rows": len(sub_df),
                "fraud_rate": float(sub_df["isFraud"].mean())
            })
            mlflow.log_metrics(summary_metrics)



