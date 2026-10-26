import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "notebooks" / "reports" / "test_submission.csv"
DEFAULT_POLICY = ROOT / "src" / "artifacts" / "thresholds.json"
BUSINESS_IMPACT_PAGE = "Fraud policy simulator"
MODEL_PIPELINE = ROOT / "assets" / "modeling_pipeline_architecture.svg"
INFERENCE_PIPELINE = ROOT / "assets" / "inference_decisioning_flow.svg"
SHAP_PLOT = ROOT / "assets" / "shap_summary.png"
PREDICTION_COLUMNS = ["TransactionID", "isFraud", "decision", "TransactionAmt"]
MIN_DECLINE_PROBABILITY = 0.25

st.set_page_config(
    page_title="Vesta IEEE-CIS (Kaggle)| Fraud Decisioning",
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
    .chart-explanation {
        color: #30343b; font-size: 1.2rem; font-weight: 500; line-height: 1.55;
    }
    .metric-explanation {
        color: #30343b; font-size: 1.2rem; font-weight: 500; line-height: 1.55;
    }
    [data-testid="stVegaLiteChart"] {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
    }
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg {
        max-width: 100%;
    }
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
def load_predictions() -> pd.DataFrame:
    if not DEFAULT_PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Test predictions were not found at {DEFAULT_PREDICTIONS}."
        )
    frame = pd.read_csv(DEFAULT_PREDICTIONS, usecols=PREDICTION_COLUMNS)

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


def get_predictions() -> pd.DataFrame | None:
    try:
        return load_predictions()
    except (
        FileNotFoundError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Unable to load test predictions: {error}")
        return None


@st.cache_data(show_spinner="Calculating threshold cost curve...")
def threshold_cost_curve(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    approve_threshold: float,
    decline_threshold: float,
    sweep_threshold: str,
    chargeback_fee: float,
    review_cost: float,
    churn_cost: float,
    catch_rate: float,
    max_review_rate: float,
) -> pd.DataFrame:
    policies: list[dict[str, float]] = []
    if sweep_threshold == "P of approve":
        thresholds = np.linspace(0.005, 0.2, 17)
    else:
        thresholds = np.linspace(
            max(approve_threshold, MIN_DECLINE_PROBABILITY), 0.7, 17
        )

    for threshold in thresholds:
        current_approve_threshold = (
            float(threshold)
            if sweep_threshold == "P of approve"
            else approve_threshold
        )
        current_decline_threshold = (
            decline_threshold
            if sweep_threshold == "P of approve"
            else float(threshold)
        )
        masks, costs, review_bypasses = evaluate_policy(
            probabilities,
            amounts,
            current_approve_threshold,
            current_decline_threshold,
            chargeback_fee,
            review_cost,
            churn_cost,
            catch_rate,
        )
        review_rate = float(np.mean(masks["REVIEW"]))
        approve_rate = float(np.mean(masks["APPROVE"]))
        fraud_loss = costs["Expected residual fraud loss"]
        staffing_cost = costs["Review staffing OpEx"]
        churn_friction = costs["Decline churn friction"]
        policies.append(
            {
                "p_approve": current_approve_threshold,
                "p_decline": current_decline_threshold,
                "total_cost": fraud_loss + staffing_cost + churn_friction,
                "fraud_loss": fraud_loss,
                "opex_cost": staffing_cost,
                "churn_cost": churn_friction,
                "review_rate": review_rate,
                "approve_rate": approve_rate,
                "review_bypassed_approve": float(
                    review_bypasses["approve"]
                ),
                "review_bypassed_decline": float(
                    review_bypasses["decline"]
                ),
                "is_feasible": float(review_rate <= max_review_rate),
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
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int]]:
    approve = probabilities < t_approve
    decline = probabilities >= max(t_decline, MIN_DECLINE_PROBABILITY)
    review = (~approve) & (~decline)

    expected_approve_cost = probabilities * (amounts + chargeback_fee)
    expected_decline_cost = (1.0 - probabilities) * churn_cost
    expected_review_cost = (
        review_cost
        + probabilities * (1.0 - catch_rate) * (amounts + chargeback_fee)
    )
    review_bypass_approve = review & (
        (expected_approve_cost < expected_review_cost)
        & (
            (probabilities < MIN_DECLINE_PROBABILITY)
            | (expected_approve_cost <= expected_decline_cost)
        )
    )
    review_bypass_decline = review & (
        (probabilities >= MIN_DECLINE_PROBABILITY)
        & (expected_decline_cost < expected_review_cost)
        & (expected_decline_cost < expected_approve_cost)
    )
    approve |= review_bypass_approve
    decline |= review_bypass_decline
    review &= ~(review_bypass_approve | review_bypass_decline)

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
    review_bypasses = {
        "approve": int(np.sum(review_bypass_approve)),
        "decline": int(np.sum(review_bypass_decline)),
    }
    return masks, costs, review_bypasses


def show_summary() -> None:
    show_hero(
        "Fraud decisioning system",
        "A chronological machine-learning workflow that balances fraud loss against customer friction.",
    )

    st.subheader("Business problem")
    st.write(
        "Vesta protects telecom and mobile-commerce payments.\n\n"
        "**Business goal:** estimate the probability that a transaction is fraudulent, "
        "then route it in a way that limits merchant losses without disrupting "
        "legitimate customers."
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

    st.subheader("Final model selection")
    st.write(
        "LightGBM was selected for serving, while the DLRM model and a "
        "rank-normalized blend are evaluated offline. The streaming test workflow "
        "scores transactions in chronological chunks, rebuilds causal features, "
        "and routes each probability to approve, review, or decline."
    )
    st.markdown(
        'See business impact of the developed model and Bayes theory based '
        '<a href="?page=business-impact" target="_self">sensitivity analysis here</a>.',
        unsafe_allow_html=True,
    )
    st.info(
        "The test-set business figures are probability-weighted estimates, not "
        "confirmed fraud outcomes. Review and decline thresholds should be "
        "recalibrated against operational, chargeback, and customer-churn costs."
    )


def show_model_and_shap() -> None:
    show_hero(
        "Model performance & explainability",
        "",
    )

    st.subheader("Training results")
    st.markdown(
        '<p class="metric-explanation">PR-AUC is especially informative for rare '
        'fraud events; ROC-AUC measures ranking quality across both classes.</p>',
        unsafe_allow_html=True,
    )
    metrics = pd.DataFrame(
        [
            ("Logistic regression — expanding window", 0.82788, 0.39200),
            ("LightGBM — expanding window", 0.91670, 0.59896),
            ("LightGBM — chronological 80/20", 0.94272, 0.66854),
            ("DLRM — 5-fold expanding-window OOF", 0.87440, 0.53673),
        ],
        columns=["Model / evaluation", "ROC-AUC", "PR-AUC"],
    )
    st.dataframe(
        metrics.style.format({"ROC-AUC": "{:.5f}", "PR-AUC": "{:.5f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "**Model selection: LightGBM for serving.** DLRM is evaluated with "
        "5-fold expanding-window out-of-fold predictions; its window differs "
        "from the chronological 80/20 LightGBM holdout, so compare the metrics "
        "with that distinction in mind. The rank-normalized LightGBM/DLRM blend "
        "is evaluated offline."
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


def show_test_pipeline() -> None:
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


def show_business_impact() -> None:
    show_hero(
        "Fraud policy simulator: What if costs change?",
        "Change the cost of fraud, manual review and customer churn, and see how the optimal approve / review / decline thresholds, review queue and total cost respond.",
    )
    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    economics = policy["economics"]
    default_approve_threshold = 0.0335
    default_decline_threshold = max(
        float(policy["thresholds"]["t_decline"]), MIN_DECLINE_PROBABILITY
    )

    if "approve_threshold_default_migrated" not in st.session_state:
        if "p_approve_threshold" not in st.session_state or np.isclose(
            st.session_state["p_approve_threshold"], 0.03
        ):
            st.session_state["p_approve_threshold"] = default_approve_threshold
        st.session_state["approve_threshold_default_migrated"] = True

    sweep_threshold = st.session_state.get("sweep_threshold", "P of decline")
    approve_threshold = st.session_state.get(
        "p_approve_threshold", default_approve_threshold
    )
    decline_threshold = st.session_state.get(
        "p_decline_threshold", default_decline_threshold
    )
    decline_threshold = min(decline_threshold, 0.7)
    st.session_state["p_decline_threshold"] = decline_threshold

    frame = get_predictions()
    if frame is None:
        return

    probabilities = frame["isFraud"].to_numpy(dtype=np.float64)
    amounts = frame["TransactionAmt"].to_numpy(dtype=np.float64)
    review_cost = st.session_state.get(
        "review_cost", float(economics["manual_review_cost"])
    )
    chargeback_fee = st.session_state.get(
        "chargeback_fee", float(economics["chargeback_fee"])
    )
    if "churn_cost_default_migrated" not in st.session_state:
        if "sensitivity_churn_cost" not in st.session_state or np.isclose(
            st.session_state["sensitivity_churn_cost"], 13.0
        ):
            st.session_state["sensitivity_churn_cost"] = 15.0
        st.session_state["churn_cost_default_migrated"] = True
    churn_cost = st.session_state.get("sensitivity_churn_cost", 15.0)
    catch_rate = st.session_state.get(
        "catch_rate", float(economics["human_catch_rate"]) * 100.0
    ) / 100.0
    if "sensitivity_max_review_rate" not in st.session_state:
        st.session_state["sensitivity_max_review_rate"] = 13.0
    max_review_rate = st.session_state["sensitivity_max_review_rate"] / 100.0

    cost_curve = threshold_cost_curve(
        probabilities,
        amounts,
        approve_threshold,
        decline_threshold,
        sweep_threshold,
        chargeback_fee,
        review_cost,
        churn_cost,
        catch_rate,
        max_review_rate,
    )
    if cost_curve.empty:
        st.warning(
            "Unable to calculate a threshold curve: the scored transaction file "
            "must contain at least one row."
        )
        return

    feasible_curve = cost_curve.loc[cost_curve["is_feasible"].astype(bool)]
    optimal_policy = (
        feasible_curve.nsmallest(1, "total_cost").iloc[0]
        if not feasible_curve.empty
        else None
    )
    selected_policy = (
        optimal_policy
        if optimal_policy is not None
        else cost_curve.loc[cost_curve["review_rate"].idxmin()]
    )
    t_approve = float(selected_policy["p_approve"])
    t_decline = float(selected_policy["p_decline"])
    total_expected_cost = (
        float(optimal_policy["total_cost"])
        if optimal_policy is not None
        else None
    )

    x_column = "p_approve" if sweep_threshold == "P of approve" else "p_decline"
    x_title = (
        "P of approve (approval threshold)"
        if sweep_threshold == "P of approve"
        else "P of decline (decline threshold)"
    )
    x_tooltip_title = "P of approve" if sweep_threshold == "P of approve" else "P of decline"
    component_data = cost_curve.melt(
        id_vars=["p_approve", "p_decline", "is_feasible"],
        value_vars=["fraud_loss", "opex_cost", "churn_cost"],
        var_name="Cost type",
        value_name="Expected cost (€)",
    )
    component_data["Cost type"] = component_data["Cost type"].map(
        {
            "fraud_loss": "Fraud Loss",
            "opex_cost": "OpEx Cost",
            "churn_cost": "Churn Cost",
        }
    )
    x_domain_min = float(cost_curve[x_column].min())
    x_domain_max = (
        0.7
        if sweep_threshold == "P of decline"
        else float(cost_curve[x_column].max())
    )
    x_scale = alt.Scale(domain=[x_domain_min, x_domain_max], clamp=True)
    x_encoding = alt.X(
        f"{x_column}:Q",
        title=x_title,
        scale=x_scale,
        axis=alt.Axis(format=".2f", labelAngle=-35),
    )
    threshold_tooltips = [
        alt.Tooltip("p_approve:Q", title="P of approve", format=".4f"),
        alt.Tooltip("p_decline:Q", title="P of decline", format=".4f"),
        alt.Tooltip("is_feasible:Q", title="Within review-queue capacity"),
    ]
    total_cost_min = float(cost_curve["total_cost"].min())
    total_cost_max = float(cost_curve["total_cost"].max())
    if total_cost_min == total_cost_max:
        total_cost_padding = max(abs(total_cost_min) * 0.01, 1.0)
    else:
        total_cost_padding = (total_cost_max - total_cost_min) * 0.05
    infeasible_rows = cost_curve.loc[
        cost_curve["is_feasible"] == 0, x_column
    ].sort_values().to_numpy(dtype=np.float64)
    sweep_values = np.sort(cost_curve[x_column].to_numpy(dtype=np.float64))
    infeasible_bands: list[dict[str, float]] = []
    for threshold in infeasible_rows:
        sample_index = int(np.searchsorted(sweep_values, threshold))
        previous_midpoint = (
            (sweep_values[sample_index - 1] + threshold) / 2
            if sample_index > 0
            else threshold - (sweep_values[1] - threshold) / 2
        )
        next_midpoint = (
            (threshold + sweep_values[sample_index + 1]) / 2
            if sample_index < len(sweep_values) - 1
            else threshold + (threshold - sweep_values[sample_index - 1]) / 2
        )
        infeasible_bands.append(
            {
                x_column: max(previous_midpoint, x_domain_min),
                "x_end": min(next_midpoint, x_domain_max),
            }
        )
    infeasible_data = pd.DataFrame(
        infeasible_bands, columns=[x_column, "x_end"]
    )
    infeasible_region = (
        alt.Chart(infeasible_data)
        .mark_rect(color="#888", opacity=0.12)
        .encode(
            x=x_encoding,
            x2="x_end:Q",
        )
    )
    total_line = (
        alt.Chart(cost_curve)
        .mark_line(
            color="#3182f6",
            point=alt.OverlayMarkDef(filled=True, size=42),
            strokeWidth=2.5,
        )
        .encode(
            x=x_encoding,
            y=alt.Y(
                "total_cost:Q",
                title="Total Expected Cost (€)",
                scale=alt.Scale(
                    domain=[
                        total_cost_min - total_cost_padding,
                        total_cost_max + total_cost_padding,
                    ],
                    zero=False,
                    nice=False,
                ),
                axis=alt.Axis(format=",.0f"),
            ),
            order=f"{x_column}:Q",
            tooltip=[
                alt.Tooltip(f"{x_column}:Q", title=x_tooltip_title, format=".4f"),
                *threshold_tooltips,
                alt.Tooltip("total_cost:Q", title="Total expected cost (€)", format=",.2f"),
            ],
        )
    )

    total_layers = []
    if not infeasible_data.empty:
        total_layers.append(infeasible_region)
    total_layers.append(total_line)
    if optimal_policy is not None:
        optimal_x = float(selected_policy[x_column])
        optimum_rule = (
            alt.Chart(pd.DataFrame({x_column: [optimal_x]}))
            .mark_rule(strokeDash=[4, 4], color="#3182f6")
            .encode(x=x_encoding)
        )
        optimal_point = (
            alt.Chart(
                pd.DataFrame(
                    [{x_column: optimal_x, "total_cost": total_expected_cost}]
                )
            )
            .mark_point(
                shape="diamond",
                filled=True,
                color="#3182f6",
                size=360,
                stroke="white",
                strokeWidth=1.5,
            )
            .encode(
                x=x_encoding,
                y="total_cost:Q",
                tooltip=[
                    alt.Tooltip(
                        f"{x_column}:Q",
                        title=f"Optimal threshold ({x_tooltip_title})",
                        format=".4f",
                    ),
                    alt.Tooltip(
                        "total_cost:Q",
                        title="Minimum expected total cost (€)",
                        format=",.2f",
                    ),
                ],
            )
        )
        total_layers.extend([optimum_rule, optimal_point])
    if not infeasible_data.empty:
        widest_band = max(
            infeasible_bands,
            key=lambda band: band["x_end"] - band[x_column],
        )
        capacity_label = (
            alt.Chart(
                pd.DataFrame(
                    [
                        {
                            x_column: (
                                widest_band[x_column] + widest_band["x_end"]
                            ) / 2,
                            "label_y": total_cost_max + total_cost_padding * 0.7,
                        }
                    ]
                )
            )
            .mark_text(color="#666", fontSize=11, text="Exceeds review capacity")
            .encode(
                x=x_encoding,
                y=alt.Y(
                    "label_y:Q",
                    scale=alt.Scale(
                        domain=[
                            total_cost_min - total_cost_padding,
                            total_cost_max + total_cost_padding,
                        ],
                        zero=False,
                        nice=False,
                    ),
                ),
            )
        )
        total_layers.append(capacity_label)
    total_chart = alt.layer(*total_layers)

    component_colors = {
        "Fraud Loss": "#c0392b",
        "OpEx Cost": "#2c6fbb",
        "Churn Cost": "#e08e0b",
    }
    panels = []
    for component_name, component_color in component_colors.items():
        component_frame = component_data.loc[
            component_data["Cost type"] == component_name
        ]
        panel_x = (
            x_encoding
            if component_name == "Churn Cost"
            else alt.X(
                f"{x_column}:Q",
                title=x_title,
                scale=x_scale,
                axis=None,
            )
        )
        component_line = (
            alt.Chart(component_frame)
            .mark_line(
                color=component_color,
                point=alt.OverlayMarkDef(filled=True, size=30),
                strokeWidth=2.5,
            )
            .encode(
                x=panel_x,
                y=alt.Y(
                    "Expected cost (€):Q",
                    title=None,
                    scale=alt.Scale(zero=False, nice=True),
                    axis=alt.Axis(format=",.2s", title=None),
                ),
                order=f"{x_column}:Q",
                tooltip=[
                    alt.Tooltip("Cost type:N", title="Cost component"),
                    alt.Tooltip(f"{x_column}:Q", title=x_tooltip_title, format=".4f"),
                    *threshold_tooltips,
                    alt.Tooltip("Expected cost (€):Q", format=",.0f"),
                ],
            )
        )
        panel_layers = []
        if not infeasible_data.empty:
            panel_layers.append(
                alt.Chart(infeasible_data)
                .mark_rect(color="#888", opacity=0.12)
                .encode(
                    x=panel_x,
                    x2="x_end:Q",
                )
            )
        panel_layers.append(component_line)
        if optimal_policy is not None:
            panel_layers.append(
                alt.Chart(pd.DataFrame({x_column: [optimal_x]}))
                .mark_rule(strokeDash=[4, 4], color="#3182f6")
                .encode(x=panel_x)
            )
        panels.append(
            alt.layer(*panel_layers).properties(
                title=f"{component_name} (€)",
                height=90,
                width="container",
            )
        )
    component_chart = alt.vconcat(
        *panels, spacing=8, title="What drives the total cost?"
    ).resolve_scale(y="independent")
    chart = alt.vconcat(
        total_chart.properties(
            title="Total Expected Cost (€)",
            height=220,
            width="container",
        ),
        component_chart,
        spacing=20,
    ).resolve_scale(y="independent")
    total_cost_range_percent = (
        (total_cost_max - total_cost_min) / total_cost_min * 100
    )
    total_cost_range_caption = (
        f"Total cost varies by only {total_cost_range_percent:.1f}% across this sweep "
        f"(€{total_cost_min:,.0f} – €{total_cost_max:,.0f}). "
        "The axis is zoomed in to show the shape."
    )

    
    st.subheader("Finding the cheapest Approve / Decline thresholds (6-month test set)")
    st.markdown(
        f'<p class="chart-explanation">Each transaction is approved, reviewed or declined, '
        f'whichever has the lowest expected cost (Bayes decision theory). '
        f'<p class="chart-explanation">This chart sweeps {sweep_threshold} while holding the other threshold '
        "fixed, and finds the minimum expected total cost under the selected "
        'business constraints. The diamond marks the lowest-cost feasible point.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("📘 How does this simulator work?", expanded=False):
            st.markdown(
                """
                **The model gives each transaction a fraud probability *p*.** The simulator
                chooses the action with the lowest *expected cost* (Bayes decision theory).
                With exposure **F = amount + chargeback fee**:
    
                | Action | Expected cost | Meaning |
                |---|---|---|
                | ✅ Approve | p × F | If it is fraud, we lose the amount and the fee |
                | 🔍 Review | audit cost + p × (1 − catch rate) × F | We always pay the audit; fraud is lost only if the reviewer misses it |
                | ⛔ Decline | (1 − p) × churn cost | If it is legitimate, we lose a customer |
    
                **Break-even probabilities** (where two actions cost the same):
                - Approve → Review at **p = audit cost / (catch rate × F)**
                - Review → Decline at **p = (churn − audit) / (churn + (1 − catch rate) × F)**
    
                The two sliders for *P of approve* / *P of decline* are global guardrails.
                Transactions in the review band are routed straight to approve or decline when
                that is cheaper. The **◆** marks the cheapest threshold that keeps the review
                queue within capacity.
    
                **Try this:** raise *churn cost* (fewer declines), raise *audit cost* (smaller
                review band), lower *queue capacity* (forces more automatic decisions).
                Costs are computed from predicted probabilities, not observed fraud labels.
                """
            )
            ex_amt = st.number_input(
                "Example transaction amount (€)",
                min_value=1.0, max_value=5000.0,
                value=float(np.median(amounts)), step=10.0,
            )
            F = ex_amt + chargeback_fee
            p_ad = churn_cost / (F + churn_cost)
            p_a = review_cost / (catch_rate * F) if catch_rate > 0 else float("inf")
            p_d = (churn_cost - review_cost) / (churn_cost + (1 - catch_rate) * F)
            if p_a < p_d:
                st.info(
                    f"For a €{ex_amt:,.0f} transaction: approve if p < **{p_a:.3f}**, "
                    f"review between **{p_a:.3f}** and **{p_d:.3f}**, decline above **{p_d:.3f}**."
                )
            else:
                st.info(
                    f"For a €{ex_amt:,.0f} transaction review never pays off with these "
                    f"costs: approve if p < **{p_ad:.3f}**, otherwise decline."
                )
    

    if optimal_policy is not None:
        summary_columns = st.columns(3)
        summary_columns[0].metric(
            f"Optimal threshold ({x_tooltip_title})",
            f"{float(selected_policy[x_column]):.2f}",
        )
        summary_columns[1].metric(
            "Minimum expected total cost",
            f"€{total_expected_cost:,.2f}",
        )
        summary_columns[2].metric(
            "P of approve / P of decline",
            f"{t_approve:.2f} / {t_decline:.2f}",
        )
    st.altair_chart(chart, width="stretch")
    st.caption(total_cost_range_caption)
    st.caption(
        "Each panel has its own scale. Shaded areas exceed the review-queue capacity, "
        "and the diamond is the cheapest feasible threshold. Raising the decline "
        "threshold means fewer declines: churn cost falls while fraud loss rises."
    )

    st.subheader("Sensitivity parameters")
    st.selectbox(
        "Threshold to sweep on the x-axis",
        ["P of decline", "P of approve"],
        key="sweep_threshold",
    )
    if sweep_threshold == "P of decline":
        st.slider(
            "P of approve",
            min_value=0.005,
            max_value=0.2,
            value=default_approve_threshold,
            step=0.0001,
            format="%.4f",
            key="p_approve_threshold",
            help="Fixed approval threshold while P of decline is swept.",
        )
    else:
        st.slider(
            "P of decline",
            min_value=MIN_DECLINE_PROBABILITY,
            max_value=0.7,
            value=default_decline_threshold,
            step=0.01,
            format="%.2f",
            key="p_decline_threshold",
            help="Fixed decline threshold while P of approve is swept.",
        )
    if st.session_state["sensitivity_max_review_rate"] > 16.0:
        st.session_state["sensitivity_max_review_rate"] = 16.0
    parameter_columns = st.columns(2)
    with parameter_columns[0]:
        st.slider(
            "Audit Cost per Review (€)",
            min_value=0.5,
            max_value=100.5,
            value=float(economics["manual_review_cost"]),
            step=1.0,
            key="review_cost",
        )
        st.slider(
            "Customer Insult / Churn Cost (€)",
            min_value=0.0,
            max_value=500.0,
            value=15.0,
            step=1.0,
            key="sensitivity_churn_cost",
        )
        st.slider(
            "Max Review Queue Capacity (%)",
            min_value=0.0,
            max_value=16.0,
            value=15.0,
            step=1.0,
            key="sensitivity_max_review_rate",
        )
    with parameter_columns[1]:
        st.slider(
            "Chargeback Fee (€)",
            min_value=0.0,
            max_value=100.0,
            value=float(economics["chargeback_fee"]),
            step=1.0,
            key="chargeback_fee",
        )
        st.slider(
            "Human Reviewer Catch Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(economics["human_catch_rate"]) * 100.0,
            step=1.0,
            key="catch_rate",
        )
    st.caption(
        "The default scored test data covers six months. The threshold sweep is "
        "evaluated against these transactions. "
        "Cost estimates use predicted probabilities rather than observed fraud labels."
    )

    st.subheader("Result metrics")
    minimum_review_rate = float(cost_curve["review_rate"].min())
    minimum_queue_capacity = int(np.ceil(minimum_review_rate * 100))
    minimum_queue_policy = cost_curve.loc[cost_curve["review_rate"].idxmin()]
    if optimal_policy is None:
        st.warning(
            f"No threshold on this sweep satisfies the {max_review_rate:.0%} "
            f"review-queue capacity. The lowest review rate is "
            f"{minimum_review_rate:.2%} at P of approve "
            f"{float(minimum_queue_policy['p_approve']):.2f} and P of decline "
            f"{float(minimum_queue_policy['p_decline']):.2f}. Increase queue "
            f"capacity to at least {minimum_queue_capacity}% or adjust the fixed "
            "threshold / sweep range."
        )
        st.dataframe(
            cost_curve[
                [
                    "p_approve",
                    "p_decline",
                    "total_cost",
                    "review_rate",
                    "review_bypassed_approve",
                    "review_bypassed_decline",
                    "is_feasible",
                ]
            ].rename(
                columns={
                    "p_approve": "P of approve",
                    "p_decline": "P of decline",
                    "total_cost": "Expected total cost (€)",
                    "review_rate": "Review queue rate",
                    "review_bypassed_approve": "Directly approved instead of reviewed",
                    "review_bypassed_decline": "Directly declined instead of reviewed",
                    "is_feasible": "Feasible",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    masks, cost_breakdown, review_bypasses = evaluate_policy(
        probabilities,
        amounts,
        t_approve,
        t_decline,
        chargeback_fee,
        review_cost,
        churn_cost,
        catch_rate,
    )
    action_counts = {
        action: int(np.sum(mask)) for action, mask in masks.items()
    }
    total_transactions = len(frame)
    review_rate = action_counts["REVIEW"] / total_transactions
    decline_rate = action_counts["DECLINE"] / total_transactions
    approved_rate = action_counts["APPROVE"] / total_transactions
    total_euros_declined = float(np.sum(amounts[masks["DECLINE"]]))
    total_euros_reviewed = float(np.sum(amounts[masks["REVIEW"]]))
    review_expense = cost_breakdown["Review staffing OpEx"]
    all_approve_fraud_loss = float(
        np.sum(probabilities * (amounts + chargeback_fee))
    )
    expected_residual_fraud_loss = cost_breakdown["Expected residual fraud loss"]
    estimated_savings = all_approve_fraud_loss - expected_residual_fraud_loss
    savings_per_transaction = estimated_savings / total_transactions
    estimated_total_cost = (
        expected_residual_fraud_loss
        + review_expense
        + cost_breakdown["Decline churn friction"]
    )
    policy_roi = estimated_savings / review_expense if review_expense else None
    metrics = pd.DataFrame(
        [
            ("Total transactions", f"{total_transactions:,}"),
            ("P of approve", f"{t_approve:.4f}"),
            ("P of decline", f"{t_decline:.4f}"),
            (
                "Transactions approved instead of reviewed",
                f"{review_bypasses['approve']:,}",
            ),
            (
                "Transactions declined instead of reviewed",
                f"{review_bypasses['decline']:,}",
            ),
            ("Auto-approve rate", f"{approved_rate:.2%}"),
            ("Manual-review rate", f"{review_rate:.2%}"),
            ("Hard-decline rate", f"{decline_rate:.2%}"),
            ("Total transaction value declined (€)", f"€{total_euros_declined:,.2f}"),
            ("Total transaction value routed to review (€)", f"€{total_euros_reviewed:,.2f}"),
            ("Manual-review operational cost (€)", f"€{review_expense:,.2f}"),
            ("Estimated fraud-loss savings vs. approving all (€)", f"€{estimated_savings:,.2f}"),
            ("Potential expected portfolio cost (€)", f"€{estimated_total_cost:,.2f}"),
            (
                "Expected residual fraud loss (€)",
                f"€{expected_residual_fraud_loss:,.2f}",
            ),
            (
                "Churn friction loss (€)",
                f"€{cost_breakdown['Decline churn friction']:,.2f}",
            ),
            (
                "Policy ROI per review euro",
                f"{policy_roi:,.2f}" if policy_roi is not None else "N/A",
            ),
        ],
        columns=["Metric", "Result"],
    )
    st.dataframe(metrics, width="stretch", hide_index=True)

    st.subheader("Key takeaways")
    takeaways = [
        f"Only transactions with fraud probability at or above "
        f"{MIN_DECLINE_PROBABILITY:.0%} can be declined (to avoid mass "
        "auto-declines of legitimate users).",
        f"The selected policy has a {decline_rate:.2%} hard-decline rate and a "
        f"{review_rate:.2%} manual-review rate. "
        f"{review_bypasses['approve']:,} transactions are approved and "
        f"{review_bypasses['decline']:,} are declined instead of reviewed "
        "because those actions have lower expected cost.",
        f"Estimated fraud-loss savings average €{savings_per_transaction:,.2f} "
        f"per transaction versus approving every transaction. This is the "
        "estimated fraud loss avoided through declines and successful reviews, "
        "averaged across the six-month test dataset.",
        f"Potential expected portfolio cost is €{estimated_total_cost:,.2f}, "
        f"including €{expected_residual_fraud_loss:,.2f} residual fraud loss, "
        f"€{review_expense:,.2f} review operating cost, and "
        f"€{cost_breakdown['Decline churn friction']:,.2f} estimated churn cost.",
        (
            f"Policy ROI is {policy_roi:,.2f} euros of estimated fraud-loss "
            "savings per euro spent on reviews."
            if policy_roi is not None
            else "Policy ROI per review euro is unavailable because review "
            "operating cost is zero."
        ),
    ]
    if optimal_policy is None:
        takeaways.append(
            f"No threshold meets the {max_review_rate:.0%} queue limit; the "
            f"lowest review rate is {minimum_review_rate:.2%}. Increase queue "
            f"capacity to at least {minimum_queue_capacity}% or adjust the fixed "
            "threshold / sweep range."
        )
    else:
        takeaways.append(
            f"The minimum-expected-cost feasible policy uses P of approve "
            f"{t_approve:.2f} and P of decline {t_decline:.2f}."
        )
    st.markdown("\n".join(f"- {takeaway}" for takeaway in takeaways))


with st.sidebar:
    st.title("Vesta IEEE-CIS | Fraud Decisioning")
    page = st.radio(
        "Report sections",
        [
            BUSINESS_IMPACT_PAGE,
            "Business problem and model pipeline",
            "Model & SHAP",
            "Testing pipeline",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p>Competition source: <a href="https://www.kaggle.com/competitions/ieee-fraud-detection" target="_blank" rel="noopener noreferrer">Kaggle</a></p>',
        unsafe_allow_html=True,
    )

if page == "Business problem and model pipeline":
    show_summary()
elif page == "Model & SHAP":
    show_model_and_shap()
elif page == "Testing pipeline":
    show_test_pipeline()
else:
    show_business_impact()
