from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "notebooks" / "reports" / "test_submission.csv"
MODEL_PIPELINE = ROOT / "modeling_pipeline_architecture.svg"
INFERENCE_PIPELINE = ROOT / "inference_decisioning_flow.svg"
SHAP_PLOT = ROOT / "assets" / "shap_summary.png"
PREDICTION_COLUMNS = ["TransactionID", "isFraud", "decision", "TransactionAmt"]

st.set_page_config(
    page_title="Vesta | Fraud Decisioning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color: #17243a; }
    [data-testid="stSidebar"] { background: #102238; }
    [data-testid="stSidebar"] * { color: #eef4fb; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        color: #f6fbff; background: linear-gradient(115deg, #102b46, #176b72);
    }
    .hero h1 { margin: 0 0 .35rem 0; color: #fff; font-size: 2.2rem; }
    .hero p { margin: 0; color: #dcecf0; font-size: 1rem; }
    .panel {
        padding: 1rem 1.15rem; border: 1px solid #e1e8f0; border-radius: 14px;
        background: #fff; min-height: 120px;
    }
    .panel h3 { margin-top: 0; color: #173653; }
    .muted { color: #62748a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading scored test transactions...")
def load_predictions(file_bytes: bytes | None = None) -> pd.DataFrame:
    if file_bytes is None:
        if not DEFAULT_PREDICTIONS.is_file():
            raise FileNotFoundError(
                f"Test predictions were not found at {DEFAULT_PREDICTIONS}."
            )
        frame = pd.read_csv(DEFAULT_PREDICTIONS, usecols=PREDICTION_COLUMNS)
    else:
        frame = pd.read_csv(
            BytesIO(file_bytes),
            usecols=PREDICTION_COLUMNS,
        )

    if frame.empty:
        raise ValueError("The scored test CSV must contain at least one transaction.")
    frame["isFraud"] = pd.to_numeric(frame["isFraud"], errors="raise")
    frame["TransactionAmt"] = pd.to_numeric(
        frame["TransactionAmt"], errors="raise"
    )
    if not frame["isFraud"].between(0, 1).all():
        raise ValueError("The isFraud column must contain probabilities from 0 to 1.")
    if frame["TransactionAmt"].isna().any():
        raise ValueError("TransactionAmt contains missing values.")
    return frame


def show_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_diagram(path: Path, height: int) -> None:
    if not path.is_file():
        st.warning(f"Pipeline diagram not found: {path.name}")
        return
    svg = path.read_text(encoding="utf-8")
    st.iframe(
        "<html><head><style>"
        "html,body{margin:0;padding:0}svg{display:block;width:100%;height:auto}"
        "</style></head><body>"
        f"{svg}</body></html>",
        width="stretch",
        height=height,
    )


def get_predictions(uploaded_file) -> pd.DataFrame | None:
    try:
        if uploaded_file is None:
            return load_predictions()
        return load_predictions(uploaded_file.getvalue())
    except (
        FileNotFoundError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Unable to load test predictions: {error}")
        return None


def show_summary() -> None:
    show_hero(
        "Fraud decisioning",
        "A chronological machine-learning workflow that balances fraud loss against customer friction.",
    )

    st.subheader("Business problem")
    st.write(
        "Vesta protects telecom and mobile-commerce payments. The business goal is "
        "to estimate the probability that a transaction is fraudulent, then route "
        "it in a way that limits merchant losses without disrupting legitimate customers."
    )

    left, middle, right = st.columns(3)
    with left:
        st.markdown(
            '<div class="panel"><h3>Missed fraud</h3><p class="muted">'
            "Undetected fraud can cause chargebacks, fees, lost transaction value, "
            "and reputational damage.</p></div>",
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            '<div class="panel"><h3>False declines</h3><p class="muted">'
            "Overly aggressive controls reject legitimate customers and put "
            "revenue and retention at risk.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="panel"><h3>Decision objective</h3><p class="muted">'
            "Minimize expected fraud loss while keeping approval friction and "
            "manual-review costs under control.</p></div>",
            unsafe_allow_html=True,
        )

    st.subheader("End-to-end modeling pipeline")
    show_diagram(MODEL_PIPELINE, height=710)

    st.subheader("Executive summary")
    st.write(
        "LightGBM was selected as the final model. It led the tested models on "
        "chronological PR-AUC, while the small ROC-AUC lift from adding a neural "
        "network did not improve PR-AUC. The streaming test workflow scores "
        "transactions in chronological chunks, rebuilds causal features, and "
        "routes each probability to approve, review, or decline."
    )
    st.info(
        "The test-set business figures are probability-weighted estimates, not "
        "confirmed fraud outcomes. Review and decline thresholds should be "
        "recalibrated against operational, chargeback, and customer-churn costs."
    )


def show_model_and_shap() -> None:
    show_hero(
        "Model performance & explainability",
        "Chronological validation and a report-grounded view of the signals behind fraud scores.",
    )

    st.subheader("Training results")
    st.caption(
        "PR-AUC is especially informative for rare fraud events; ROC-AUC measures "
        "ranking quality across both classes."
    )
    metrics = pd.DataFrame(
        [
            ("Logistic regression — expanding window", 0.82788, 0.39200),
            ("LightGBM — expanding window", 0.91670, 0.59896),
            ("LightGBM — chronological 80/20", 0.94272, 0.66854),
            ("Neural network — chronological 80/20", 0.88958, 0.50286),
            ("Rank blend (95% LightGBM / 5% NN)", 0.94293, 0.66185),
        ],
        columns=["Model / evaluation", "ROC-AUC", "PR-AUC"],
    )
    st.dataframe(
        metrics.style.format({"ROC-AUC": "{:.5f}", "PR-AUC": "{:.5f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        **Model selection: LightGBM.** The rank blend improved ROC-AUC by just
        0.00021 over LightGBM but had lower PR-AUC. For the rare-fraud use case,
        the added neural-network complexity did not justify selecting the blend.
        """
    )

    st.subheader("SHAP: what moves risk scores")
    st.caption(
        "These are the qualitative SHAP patterns reported in the analysis; "
        "feature contribution indicates model influence, not causation."
    )
    if SHAP_PLOT.is_file():
        st.image(
            SHAP_PLOT,
            caption=(
                "Top 25 features. SHAP values show each feature's impact on model "
                "output; color represents feature value."
            ),
        )
    else:
        st.warning(f"SHAP summary figure not found: {SHAP_PLOT}")

    first, second = st.columns(2)
    with first:
        st.markdown(
            '<div class="panel"><h3>Transaction behavior</h3><p class="muted">'
            "Higher transaction amounts and elevated activity or velocity counters "
            "tend to push model scores toward fraud.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel"><h3>Historical risk encoding</h3><p class="muted">'
            "Categories with higher historical fraud rates contribute more risk "
            "through their target-encoded features.</p></div>",
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            '<div class="panel"><h3>Card & identity history</h3><p class="muted">'
            "Card age and reconstructed-identity or card frequencies help capture "
            "repeat behavior and entity-level risk.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel"><h3>Vesta signals</h3><p class="muted">'
            "Selected V-features also act as fraud triggers; their contribution "
            "is interpreted alongside the feature-availability patterns.</p></div>",
            unsafe_allow_html=True,
        )


def show_test_pipeline() -> pd.DataFrame | None:
    show_hero(
        "Testing pipeline",
        "Stateful, time-ordered inference using frozen training artifacts and a three-tier policy.",
    )
    show_diagram(INFERENCE_PIPELINE, height=570)

    st.markdown(
        """
        **Inference sequence:** join transaction and identity records → sort by
        transaction time → process in chunks → seed DuckDB with recent training
        history → calculate causal 24-hour user features → apply saved encodings
        and feature schema → score with LightGBM → emit a decision.
        """
    )
    st.caption(
        "The default view uses the saved scored test output. Upload another scored "
        "CSV with TransactionID, isFraud (probability), decision, and TransactionAmt "
        "to inspect it. This page does not retrain the model or score raw uploads."
    )
    uploaded_file = st.file_uploader(
        "Optional: inspect a scored test CSV",
        type=["csv"],
        help="Expected columns: TransactionID, isFraud, decision, TransactionAmt.",
    )
    frame = get_predictions(uploaded_file)
    if frame is None:
        return None

    st.subheader("Scored test output")
    st.metric("Transactions scored", f"{len(frame):,}")
    if "decision" in frame:
        counts = frame["decision"].value_counts().reindex(
            ["APPROVE", "REVIEW", "DECLINE"], fill_value=0
        )
        chart_data = counts.rename_axis("Decision").rename("Transactions").reset_index()
        st.bar_chart(chart_data, x="Decision", y="Transactions")
    st.dataframe(frame.head(20), width="stretch", hide_index=True)
    return frame


def show_business_impact() -> None:
    show_hero(
        "Business impact",
        "Explore how review and decline thresholds change estimated exposure and operating cost.",
    )
    st.caption(
        "Starting policy from the report: review at 0.05, decline at 0.35; "
        "manual review cost 2.50 and chargeback fee 15.00. Amounts and costs use "
        "the transaction dataset's currency units."
    )

    uploaded_file = st.file_uploader(
        "Optional: use a different scored test CSV",
        type=["csv"],
        key="impact_upload",
        help="Expected columns: TransactionID, isFraud (probability), decision, TransactionAmt.",
    )
    frame = get_predictions(uploaded_file)
    if frame is None:
        return

    left, right = st.columns(2)
    with left:
        review_threshold = st.number_input(
            "Review from probability",
            min_value=0.0,
            max_value=0.99,
            value=0.05,
            step=0.01,
            format="%.2f",
        )
    with right:
        decline_threshold = st.number_input(
            "Decline from probability",
            min_value=0.01,
            max_value=1.0,
            value=0.35,
            step=0.01,
            format="%.2f",
        )
    cost_left, cost_right = st.columns(2)
    with cost_left:
        review_cost = st.number_input(
            "Cost per manual review",
            min_value=0.0,
            value=2.50,
            step=0.50,
        )
    with cost_right:
        chargeback_fee = st.number_input(
            "Chargeback fee",
            min_value=0.0,
            value=15.00,
            step=1.00,
        )

    if review_threshold >= decline_threshold:
        st.warning("The decline threshold must be higher than the review threshold.")
        return

    probabilities = frame["isFraud"].to_numpy()
    amounts = frame["TransactionAmt"].to_numpy()
    actions = np.select(
        [
            probabilities < review_threshold,
            probabilities < decline_threshold,
        ],
        ["APPROVE", "REVIEW"],
        default="DECLINE",
    )
    action_counts = pd.Series(actions).value_counts().reindex(
        ["APPROVE", "REVIEW", "DECLINE"], fill_value=0
    )
    review_count = int(action_counts["REVIEW"])
    decline_count = int(action_counts["DECLINE"])
    approved_mask = actions == "APPROVE"
    declined_mask = actions == "DECLINE"
    expected_loss = float(
        np.sum(probabilities[approved_mask] * (amounts[approved_mask] + chargeback_fee))
    )
    prevented_loss = float(
        np.sum(probabilities[declined_mask] * (amounts[declined_mask] + chargeback_fee))
    )
    total_expected_fraud_cost = prevented_loss + expected_loss
    prevented_share = (
        prevented_loss / total_expected_fraud_cost if total_expected_fraud_cost else 0.0
    )
    review_spend = review_count * review_cost

    st.subheader("Estimated operating outcomes")
    columns = st.columns(4)
    columns[0].metric(
        "Auto-approve",
        f"{action_counts['APPROVE'] / len(frame):.2%}",
        f"{action_counts['APPROVE']:,} transactions",
    )
    columns[1].metric(
        "Manual review",
        f"{review_count / len(frame):.2%}",
        f"{review_count:,} · cost {review_spend:,.2f}",
    )
    columns[2].metric(
        "Hard decline",
        f"{decline_count / len(frame):.2%}",
        f"{decline_count:,} transactions",
    )
    columns[3].metric(
        "Expected loss prevented",
        f"{prevented_share:.1%}",
        f"{prevented_loss:,.2f} probability-weighted",
    )

    outcome_data = pd.DataFrame(
        {
            "Transactions": action_counts,
            "Share": action_counts / len(frame),
        }
    )
    chart_data = outcome_data.rename_axis("Decision").reset_index()
    st.bar_chart(chart_data, x="Decision", y="Share")
    st.markdown(
        f"""
        - **Expected fraud exposure in auto-approved transactions:** {expected_loss:,.2f}
        - **Probability-weighted loss for declined transactions:** {prevented_loss:,.2f}
        - **Manual-review operating cost:** {review_spend:,.2f}

        These are model-based estimates: the saved test output has no observed
        fraud labels. Review outcomes are not counted as prevented loss here.
        """
    )
    st.dataframe(
        outcome_data.style.format({"Share": "{:.2%}"}),
        width="stretch",
    )

    st.subheader("Business recommendations")
    st.write(
        "Use threshold tuning as a cost-policy decision, not a model-score exercise "
        "alone. Calibrate against chargeback loss, review capacity, and the cost of "
        "rejecting legitimate customers; consider a minimum transaction amount "
        "for manual review."
    )


with st.sidebar:
    st.title("Vesta | Fraud")
    page = st.radio(
        "Report sections",
        [
            "Executive summary",
            "Model & SHAP",
            "Testing pipeline",
            "Business impact",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("IEEE-CIS fraud detection report")

if page == "Executive summary":
    show_summary()
elif page == "Model & SHAP":
    show_model_and_shap()
elif page == "Testing pipeline":
    show_test_pipeline()
else:
    show_business_impact()
