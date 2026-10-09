import json
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "notebooks" / "reports" / "test_submission.csv"
DEFAULT_POLICY = ROOT / "src" / "artifacts" / "thresholds.json"
BUSINESS_IMPACT_PAGE = "Business impact - What-If sensitivity analysis simulator"
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


@st.cache_data(show_spinner="Evaluating feasible threshold combinations...")
def threshold_cost_surface(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    chargeback_fee: float,
    review_cost: float,
    churn_cost: float,
    catch_rate: float,
    max_review_rate: float,
) -> pd.DataFrame:
    expected_approve_cost = probabilities * (amounts + chargeback_fee)
    expected_decline_cost = (1.0 - probabilities) * churn_cost
    cannot_review = (amounts + chargeback_fee) <= review_cost
    prefer_approve = expected_approve_cost <= expected_decline_cost
    total_count = len(probabilities)
    policies: list[dict[str, float]] = []

    for approve_threshold in np.linspace(0.005, 0.2, 30):
        approve_base = probabilities < approve_threshold
        for decline_threshold in np.linspace(0.2, 0.45, 30):
            if decline_threshold <= approve_threshold:
                continue

            decline_base = probabilities >= decline_threshold
            review_base = (~approve_base) & (~decline_base)
            review_bypass = review_base & cannot_review
            review = review_base & ~cannot_review
            approve = approve_base | (review_bypass & prefer_approve)
            decline = decline_base | (review_bypass & ~prefer_approve)
            review_rate = float(np.sum(review) / total_count)
            approve_rate = float(np.sum(approve) / total_count)
            if review_rate > max_review_rate or approve_rate < 0.85:
                continue

            fraud_loss = float(
                np.sum(expected_approve_cost[approve])
                + np.sum(
                    probabilities[review]
                    * (1.0 - catch_rate)
                    * (amounts[review] + chargeback_fee)
                )
            )
            review_staffing_cost = float(np.sum(review) * review_cost)
            churn_friction_cost = float(np.sum(expected_decline_cost[decline]))
            policies.append(
                {
                    "t_approve": float(approve_threshold),
                    "t_decline": float(decline_threshold),
                    "expected_fraud_loss": fraud_loss,
                    "review_staffing_cost": review_staffing_cost,
                    "churn_friction_cost": churn_friction_cost,
                    "total_expected_cost": (
                        fraud_loss + review_staffing_cost + churn_friction_cost
                    ),
                    "review_rate": review_rate,
                    "approve_rate": approve_rate,
                }
            )
    return pd.DataFrame(policies)


def evaluate_policy(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    t_approve: float,
    t_decline: float,
    chargeback_fee: float,
    review_cost: float,
    churn_cost: float,
    catch_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    approve = probabilities < t_approve
    decline = probabilities >= t_decline
    review = (~approve) & (~decline)

    cannot_review = (amounts + chargeback_fee) <= review_cost
    prefer_approve = (
        probabilities * (amounts + chargeback_fee)
        <= (1.0 - probabilities) * churn_cost
    )
    review_bypass = review & cannot_review
    review &= ~cannot_review
    approve |= review_bypass & prefer_approve
    decline |= review_bypass & ~prefer_approve

    masks = {"APPROVE": approve, "REVIEW": review, "DECLINE": decline}
    residual_fraud_loss = float(
        np.sum(probabilities[approve] * (amounts[approve] + chargeback_fee))
        + np.sum(
            probabilities[review]
            * (1.0 - catch_rate)
            * (amounts[review] + chargeback_fee)
        )
    )
    staffing_cost = float(np.sum(review) * review_cost)
    churn_friction = float(np.sum((1.0 - probabilities[decline]) * churn_cost))
    costs = {
        "Expected residual fraud loss": residual_fraud_loss,
        "Review staffing OpEx": staffing_cost,
        "Decline churn friction": churn_friction,
    }
    return masks, costs


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
        BUSINESS_IMPACT_PAGE,
        "Explore how review economics and queue capacity change the minimum-cost fraud decision policy.",
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

    probabilities = frame["isFraud"].to_numpy(dtype=np.float64)
    amounts = frame["TransactionAmt"].to_numpy(dtype=np.float64)
    chargeback_fee = st.session_state["chargeback_fee"]
    review_cost = st.session_state["review_cost"]
    churn_cost = st.session_state["churn_cost"]
    catch_rate = st.session_state["catch_rate"] / 100.0
    max_review_rate = st.session_state["max_review_rate"] / 100.0

    cost_surface = threshold_cost_surface(
        probabilities,
        amounts,
        chargeback_fee,
        review_cost,
        churn_cost,
        catch_rate,
        max_review_rate,
    )
    if cost_surface.empty:
        st.warning(
            "No feasible threshold pair was found for these assumptions. "
            "The optimizer requires at least 85% auto-approval and enforces "
            "the selected review-queue capacity."
        )
        return

    optimal_policy = cost_surface.nsmallest(1, "total_expected_cost").iloc[0]
    t_approve = float(optimal_policy["t_approve"])
    t_decline = float(optimal_policy["t_decline"])
    masks, cost_breakdown = evaluate_policy(
        probabilities,
        amounts,
        t_approve,
        t_decline,
        chargeback_fee,
        review_cost,
        churn_cost,
        catch_rate,
    )
    action_counts = pd.Series(
        {action: int(np.sum(mask)) for action, mask in masks.items()}
    )
    review_count = action_counts["REVIEW"]
    review_rate = review_count / len(frame)
    total_expected_cost = float(optimal_policy["total_expected_cost"])

    st.caption(
        "Each point is a feasible threshold pair from a 30×30 grid, subject to the "
        "selected review-queue limit and a minimum 85% auto-approval rate. Costs are "
        "estimated from predicted probabilities, not observed fraud outcomes; cost "
        "inputs and transaction amounts are treated as euros."
    )
    st.subheader("Optimal thresholds for current assumptions")
    columns = st.columns(3)
    columns[0].metric(
        "Optimal approve threshold",
        f"{t_approve:.4f}",
    )
    columns[1].metric(
        "Optimal decline threshold",
        f"{t_decline:.4f}",
    )
    columns[2].metric(
        "Minimum expected portfolio cost",
        f"€{total_expected_cost:,.2f}",
    )

    st.subheader("Total cost by approve and decline thresholds")
    chart_data = cost_surface.copy()
    chart_data["is_optimal"] = (
        (chart_data["t_approve"] == t_approve)
        & (chart_data["t_decline"] == t_decline)
    )
    chart_data["marker_size"] = np.where(chart_data["is_optimal"], 280, 55)
    threshold_points = (
        alt.Chart(chart_data)
        .mark_circle(filled=True, opacity=0.8)
        .encode(
            x=alt.X(
                "t_approve:Q",
                title="Approve threshold (Tₐ)",
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y(
                "t_decline:Q",
                title="Decline threshold (Tᵈ)",
                axis=alt.Axis(format=".2f"),
            ),
            color=alt.Color(
                "total_expected_cost:Q",
                title="Expected total cost (€)",
                scale=alt.Scale(scheme="viridis"),
            ),
            size=alt.Size(
                "marker_size:Q",
                scale=alt.Scale(domain=[55, 280], range=[55, 280]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("t_approve:Q", title="Approve threshold", format=".4f"),
                alt.Tooltip("t_decline:Q", title="Decline threshold", format=".4f"),
                alt.Tooltip(
                    "total_expected_cost:Q",
                    title="Total cost (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "expected_fraud_loss:Q",
                    title="Fraud loss (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "review_staffing_cost:Q",
                    title="Review OpEx (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "churn_friction_cost:Q",
                    title="Churn friction (€)",
                    format=",.2f",
                ),
            ],
        )
    )
    optimal_point = (
        alt.Chart(chart_data.loc[chart_data["is_optimal"]])
        .mark_point(
            shape="diamond",
            filled=True,
            color="#e63946",
            size=220,
            stroke="white",
            strokeWidth=1.5,
        )
        .encode(
            x="t_approve:Q",
            y="t_decline:Q",
            tooltip=[
                alt.Tooltip("t_approve:Q", title="Optimal approve threshold", format=".4f"),
                alt.Tooltip("t_decline:Q", title="Optimal decline threshold", format=".4f"),
                alt.Tooltip(
                    "total_expected_cost:Q",
                    title="Minimum expected total cost (€)",
                    format=",.2f",
                ),
            ],
        )
    )
    optimal_label = (
        alt.Chart(chart_data.loc[chart_data["is_optimal"]])
        .mark_text(align="left", dx=10, dy=-10, color="#b42332")
        .encode(
            x="t_approve:Q",
            y="t_decline:Q",
            text=alt.value("Optimal"),
        )
    )
    st.altair_chart(
        alt.layer(threshold_points, optimal_point, optimal_label).properties(
            height=440
        ),
        width="stretch",
    )

    st.subheader("Minimum-cost breakdown")
    st.dataframe(
        pd.DataFrame(
            {
                "Cost component": [
                    "Expected fraud loss",
                    "Review staffing OpEx",
                    "Decline churn friction",
                ],
                "Expected cost (€)": [
                    cost_breakdown["Expected residual fraud loss"],
                    cost_breakdown["Review staffing OpEx"],
                    cost_breakdown["Decline churn friction"],
                ],
            }
        ).style.format({"Expected cost (€)": "€{:,.2f}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Review-queue breakeven boundary")
    review_band = (probabilities >= t_approve) & (probabilities < t_decline)
    breakeven_bypass = review_band & (
        (amounts + chargeback_fee) <= review_cost
    )
    queue_columns = st.columns(3)
    queue_columns[0].metric(
        "Review-band candidates",
        f"{int(np.sum(review_band)):,}",
    )
    queue_columns[1].metric(
        "Routed to reviewers",
        f"{review_count:,}",
        f"{review_rate:.2%} of batch",
    )
    queue_columns[2].metric(
        "Breakeven bypass",
        f"{int(np.sum(breakeven_bypass)):,}",
        "Automatically approved or declined",
    )
    st.write(
        f"Scores from **{t_approve:.4f}** up to (but not including) "
        f"**{t_decline:.4f}** form the review band. The optimizer holds the queue "
        f"to **{review_rate:.2%}**, below the **{max_review_rate:.2%}** capacity "
        f"limit. Low-value transactions where review cost exceeds transaction "
        f"amount plus chargeback fee bypass review and are assigned to the cheaper "
        f"approve/decline action; transactions outside the band are auto-routed by "
        f"their score thresholds."
    )
    bin_edges = np.linspace(0.0, 1.0, 21)
    histogram = {
        action: np.histogram(probabilities[mask], bins=bin_edges)[0]
        for action, mask in masks.items()
    }
    probability_bands = [
        f"{lower:.2f}–{upper:.2f}"
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:])
    ]
    chart_data = pd.DataFrame(histogram, index=probability_bands)
    chart_data.index.name = "Fraud-probability band"
    st.bar_chart(chart_data, stack=True)

    outcome_data = pd.DataFrame(
        {
            "Transactions": action_counts,
            "Share": action_counts / len(frame),
        }
    )
    st.dataframe(
        outcome_data.style.format({"Share": "{:.2%}"}),
        width="stretch",
    )
    st.info(
        "This is a what-if estimate, not a realized savings forecast. Validate the "
        "assumptions and currency alignment before operational use."
    )


with st.sidebar:
    st.title("Vesta | Fraud")
    page = st.radio(
        "Report sections",
        [
            "Executive summary",
            "Model & SHAP",
            "Testing pipeline",
            BUSINESS_IMPACT_PAGE,
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if page == BUSINESS_IMPACT_PAGE:
        policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
        economics = policy["economics"]
        st.subheader("What-if assumptions")
        st.slider(
            "Audit Cost per Review (€)",
            min_value=0.5,
            max_value=100.5,
            value=float(economics["manual_review_cost"]),
            step=1.0,
            key="review_cost",
        )
        st.slider(
            "Chargeback Fee (€)",
            min_value=0.0,
            max_value=100.0,
            value=float(economics["chargeback_fee"]),
            step=1.0,
            key="chargeback_fee",
        )
        st.slider(
            "Customer Insult / Churn Cost (€)",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
            step=1.0,
            key="churn_cost",
        )
        st.slider(
            "Human Reviewer Catch Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(economics["human_catch_rate"]) * 100.0,
            step=1.0,
            key="catch_rate",
        )
        st.slider(
            "Max Review Queue Capacity (%)",
            min_value=0.0,
            max_value=25.0,
            value=8.0,
            step=1.0,
            key="max_review_rate",
        )
        st.caption("The optimizer also enforces a minimum 85% auto-approval rate.")
    st.caption("IEEE-CIS fraud detection report")

if page == "Executive summary":
    show_summary()
elif page == "Model & SHAP":
    show_model_and_shap()
elif page == "Testing pipeline":
    show_test_pipeline()
else:
    show_business_impact()
