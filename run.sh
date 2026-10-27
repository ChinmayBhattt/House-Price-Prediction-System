#!/usr/bin/env bash
# run.sh - Quickstart script for Pan-India House Price Prediction System

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi

echo "=========================================================="
echo "🏙️ Pan-India House Price Prediction System"
echo "Author: Chinmay Bhatt (24EARAD034)"
echo "Arya College of Engineering & I.T. | DMV CoreTech"
echo "Coverage: Mumbai, Delhi NCR, Bengaluru, Hyderabad, Pune, Jaipur, Kolkata, Ahmedabad + Custom Cities"
echo "=========================================================="

# Check if port 8501 is already running and terminate old instance
PID=$(lsof -ti :8501 || true)
if [ -n "$PID" ]; then
    echo "Freeing port 8501 (killing process $PID)..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
fi

echo "Starting Streamlit UI on http://localhost:8501 ..."
.venv/bin/streamlit run app.py --server.port 8501 --server.headless false
