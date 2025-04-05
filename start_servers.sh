#!/bin/sh
# Start the backend server
gunicorn --bind 0.0.0.0:$BACKEND_PORT app:app &
# Start the frontend server
streamlit run frontend.py --server.port $PORT