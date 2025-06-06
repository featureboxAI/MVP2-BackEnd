# Forecasting Backend (Deployed via GCR + Cloud Run)

A FastAPI-based backend service for processing and forecasting data. This service handles ZIP file uploads containing Excel files, processes them, and generates forecasts using time series models.

## Features

- ZIP file upload and processing
- Excel file extraction and validation
- Uploads Excel files to Google Cloud Storage (GCS)
- Time series forecasting using Holt-Winters method
- Forecast generation for multiple items
- MAPE (Mean Absolute Percentage Error) calculation
- Excel file output with forecasts and error metrics

## Project Structure

```
mvp2-BackEnd/
├── app/
│   ├── main.py              # FastAPI app and ZIP upload endpoint
│   ├── forecasting.py       # Forecasting and Excel generation logic
│   └── __init__.py
├── forecast_outputs/        # (Used locally, not in GCR)
├── service_account_key.json # GCP service credentials (used in Docker build)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── README.md
```

## 🧱 Requirements

- Python ≥ 3.8
- GCP Project with:
  - GCS bucket 
  - Service account with Storage write access
- Included Python packages:
  - `fastapi`, `pandas`, `numpy`, `statsmodels`, `scikit-learn`
  - `google-cloud-storage`, `python-multipart`, `openpyxl`

---

## ☁️ Deployment Overview (GCP Cloud Run)

1. **Build Docker image**  
   ```bash
   docker build -t gcr.io/<PROJECT-ID>/fb-ai-service .
   ```

2. **Push to Google Container Registry (GCR)**  
   ```bash
   docker push gcr.io/<PROJECT-ID>/fb-ai-service
   ```

3. **Deploy to Cloud Run**  
   ```bash
   gcloud run deploy fb-ai-service \
     --image gcr.io/<PROJECT-ID>/fb-ai-service \
     --platform managed \
     --region us-west1 \
     --allow-unauthenticated
   ```

4. **Access Swagger UI**  
   Visit:  
   ```
   https://<your-service-url>.run.app/docs
   ```
--

## 📤 API Endpoint

### `POST /upload/`

- Accepts: `multipart/form-data`
- Body: `.zip` file containing one or more `.xlsx` files
- Each Excel file should include a time series format:
  - Columns expected: `Item`, `Date`, `Value`
  - Minimum data required for forecasting: ≥12 months per item

---

## Output

After processing:
- An Excel file is generated with:
  - Forecasted values for each item
  - MAPE (Mean Absolute Percentage Error)
  - Metrics summary
- The response returns this file as a downloadable attachment

> ⚠️ Note: In Swagger UI, auto-download is **not supported**. You must click the download link to retrieve the file.


## Error Handling

The service handles various error cases:
- Invalid file types
- Corrupt ZIP files
- Missing required Excel files
- Data format issues
- GCS upload failures


## Dependencies

Key dependencies include:
- FastAPI
- pandas
- numpy
- statsmodels
- scikit-learn
- google-cloud-storage
- python-multipart

See `requirements.txt` for complete list of dependencies.