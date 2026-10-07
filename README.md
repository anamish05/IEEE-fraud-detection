# IEEE-fraud-detection

## Streamlit report

Run the report dashboard from the repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app presents the business problem, model pipeline and training results,
SHAP interpretation, chronological inference pipeline, and an interactive
business-impact threshold simulator. It uses the saved scored test output at
`notebooks/reports/test_submission.csv` by default. The optional CSV upload
expects `TransactionID`, `isFraud` (a fraud probability), `decision`, and
`TransactionAmt`; raw transaction data is not scored by the dashboard.