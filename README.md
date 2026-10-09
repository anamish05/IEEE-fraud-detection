# IEEE-fraud-detection

## Streamlit report

Run the report dashboard from the repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app presents the business problem, model pipeline and training results,
SHAP interpretation, chronological inference pipeline, and an interactive
what-if sensitivity simulator. On **Business impact - What-If sensitivity
analysis simulator**, sidebar sliders adjust audit cost, chargeback fee, churn
cost, reviewer catch rate, and maximum review-queue capacity. The simulator
evaluates feasible approve/decline threshold pairs, plots expected total cost
against both thresholds, and marks the minimum-cost pair. Cost increments are
€1, and the default queue capacity is 8%. Results are scenario estimates based
on predicted probabilities, not observed fraud outcomes. It uses the saved
scored test output at `notebooks/reports/test_submission.csv` by default. The
optional CSV upload expects `TransactionID`, `isFraud` (a fraud probability),
`decision`, and `TransactionAmt`; raw transaction data is not scored by the
dashboard.