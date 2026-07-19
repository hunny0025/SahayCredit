# SahayCredit ML Scoring Service (Active)

**Status:** ACTIVE — Integrated into the SahayCredit deployment pipeline.

This FastAPI microservice exposes two ML models as HTTP endpoints:

1. **Credit Risk Model (XGBoost)** — `POST /credit/score`
   Predicts default probability and generates a PDO-calibrated credit score (300–900) with SHAP reason codes.

2. **Fraud Detection Model (LightGBM, IEEE-CIS)** — `POST /fraud/score`
   Predicts fraud probability, calibrated against a reference distribution, with SHAP reason codes.

3. **Combined Decision** — `POST /decision`
   Applies a credit × fraud decision matrix → Approve / Manual Review / Reject.

## Running Locally

```bash
cd ml/fraud_service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Render Deployment

This service is deployed as a separate Python web service on Render (`sahaycredit-ml`).
The Node.js backend connects to it via the `ML_SERVICE_URL` environment variable.

## Model Files

| File | Purpose |
|------|---------|
| `sahaycredit_xgb.json` | XGBoost credit risk model |
| `fraud_model_ieee.txt` | LightGBM fraud detection model (IEEE-CIS) |
| `target_encoding_maps.json` | Categorical encoding maps for credit features |
| `fraud_reference_probs.npy` | Reference probability distribution for fraud score calibration |

## Known Limitations

- The fraud model was trained on IEEE-CIS (US card transactions). Domain accuracy for Indian UPI transactions is unvalidated.
- The credit model uses Home Credit features mapped from SahayCredit's behavioral signals.
- For production deployment, retraining on Indian transaction data is recommended.
