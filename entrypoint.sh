#!/bin/sh
set -e

echo " Starting FastAPI..."
echo " PYTHONPATH is: $PYTHONPATH"
echo " Port is: ${PORT:-8080}"

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
