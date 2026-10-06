import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from .duckdb_store import DuckDBStateStore
import dagshub
import mlflow
import mlflow.lightgbm
from src.config import DAGSHUB_USERNAME, DAGSHUB_REPO
from dotenv import load_dotenv

class TransactionStreamer:

    def __init__(
        self,
        model_uri: str,
        features_path: str,
        target_enc_path: str,
        freq_enc_path: str,
        train_warmup_path: str,
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

        with open("v_clusters.json", "r") as f:
            self.v_clusters=json.load(f)

        if thresholds_path:
            with open(thresholds_path, 'r') as f:
                self.policy = json.load(f)
        else:
            self.policy={'t_review':0.05, 't_decline':0.35, 'manual_review_cost':2.50, 'chargeback_fee': 15.00}

    def run_inference(self, test_trans_path: str, test_ident_path: str, output_csv: str, chunk_size: int = 50000):


        with mlflow.start_run(run_name="streaming-inference"):
            mlflow.log_params({
                "chunk_size": chunk_size,
                "feature_count": len(self.model_features)
            })




            # Create stream view
            self.store.con.execute(f"""
                CREATE VIEW test_stream_view AS 
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
                FROM read_parquet('{test_trans_path}') t
                LEFT JOIN read_parquet('{test_ident_path}') i ON t.TransactionID = i.TransactionID
                ORDER BY t.TransactionDT ASC;
            """)

            cursor = self.store.con.cursor()
            cursor.execute("SELECT * FROM test_stream_view;")

            predictions, tx_ids, actions = [], [], []
            transaction_amounts= []

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
                            -1::BIGINT AS TransactionID, Pseudo_UID, event_time, TransactionAmt, 
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
                                ORDER BY event_time, is_chunk, TransactionID
                                RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
                            ) AS uid_count_24h,
                            
                            -- Mean amount in preceding 24 hours
                            COALESCE(
                                AVG(TransactionAmt) OVER (
                                    PARTITION BY Pseudo_UID 
                                    ORDER BY event_time, is_chunk, TransactionID
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
                    LEFT JOIN user_last_tx ult ON c.PseudoID=ult.pseudouid
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

                # Predict
                X_chunk = chunk_with_features_df[self.model_features]
                preds = self.model.predict(X_chunk)

                conditions = [preds < self.policy['t_review'], (preds >= self.policy['t_review'])&(preds < self.policy['t_decline']), preds >= self.policy['t_decline'] ]

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

            # Economic parameters from policy (with fallbacks)
            cb_fee = self.policy.get("economics", {}).get("chargeback_fee", 15.0)
            review_cost = self.policy.get("economics", {}).get("manual_review_cost", 2.50)

            total_tx = len(sub_df)
            action_counts = sub_df["decision"].value_counts().to_dict()
           
            n_approved = action_counts.get("APPROVE", 0)
            n_reviewed = action_counts.get("REVIEW", 0)
            n_declined = action_counts.get("DECLINE", 0)

            # Operational queue cost incurred by staff
            total_review_expense = n_reviewed * review_cost
            # Expected Chargeback Exposure = sum of P(fraud) * (Amt + dispute_fee)
            sub_df["expected_cb_loss"] = sub_df["isFraud"] * (sub_df["TransactionAmt"] + cb_fee)

            expected_approved_loss = sub_df.loc[sub_df["decision"] == "APPROVE", "expected_cb_loss"].sum()
            prevented_fraud_exposure = sub_df.loc[sub_df["decision"] == "DECLINE", "expected_cb_loss"].sum()


            summary_metrics = {
            "total_transactions": total_tx,
            "auto_approve_rate": n_approved / total_tx,
            "manual_review_rate": n_reviewed / total_tx,
            "hard_decline_rate": n_declined / total_tx,
            "total_euros_declined": float(sub_df.loc[sub_df["decision"] == "DECLINE", "TransactionAmt"].sum()),
            "total_euros_routed_to_review": float(sub_df.loc[sub_df["decision"] == "REVIEW", "TransactionAmt"].sum()),
            "manual_review_operational_cost": float(total_review_expense),
            "expected_fraud_loss": float(expected_approved_loss),
            "prevented_fraud_loss": float(prevented_fraud_exposure),
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
            mlflow.log_artifact("configs/thresholds.json")



