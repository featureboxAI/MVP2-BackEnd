#!/bin/sh
set -e

echo " Starting FastAPI..."
echo " PYTHONPATH is: $PYTHONPATH"
echo " Port is: ${PORT:-8080}"
echo " Current directory: $(pwd)"
echo " Files in current directory: $(ls -la)"

# Test Python imports
echo "Testing Python imports..."
python -c "import google.auth; print('google-auth version:', google.auth.__version__)"
python -c "import app.main; print('app.main import successful')"

# Check for service account file (optional for Cloud Run)
echo "Checking for service account file..."
if [ -f "service_account_key.json" ]; then
    echo "✓ service_account_key.json found (optional for Cloud Run)"
elif [ -f "/app/service_account_key.json" ]; then
    echo "✓ service_account_key.json found in /app/ (optional for Cloud Run)"
else
    echo "ℹ service_account_key.json not found (using Cloud Run default credentials)"
fi

# Start the FastAPI server
echo "Starting uvicorn server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level debug

