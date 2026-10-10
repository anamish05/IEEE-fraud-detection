# IEEE-fraud-detection

## Streamlit report

Run the report dashboard from the repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app presents the business problem, model pipeline and training results,
SHAP interpretation, chronological inference pipeline, and an interactive
what-if sensitivity simulator. On **Business impact - Sensitivity analysis**,
the default scored test data covers six months. Users can sweep either P of
approve or P of decline while setting the other threshold as a parameter. The
P of decline sweep runs up to 0.70, and no transaction can be declined below a
25% fraud probability. A dual chart shows tightly scaled total expected cost
above and fraud-loss, review operating, and churn components below, with a
larger diamond marking the minimum-cost feasible threshold.
Sensitivity controls use a default churn cost of €15 and review-queue capacity
of 13% (maximum 16%); transactions are routed directly to approve or decline
instead of review when that action has lower expected cost. Result metrics and key
takeaways include review ROI, rates, estimated savings, and potential portfolio
cost. Results use predicted probabilities rather than observed fraud outcomes.
The dashboard uses the saved scored test output at
`notebooks/reports/test_submission.csv`; it does not accept alternate test sets
or score raw transaction data.