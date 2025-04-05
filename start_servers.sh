#!/bin/sh
streamlit run frontend.py --server.port 8000
# Log the start of the script
# echo "Starting start_servers.sh script" | logger

# # Start the backend server and log its output
# echo "Starting backend server on port $BACKEND_PORT" | logger
# gunicorn --bind 0.0.0.0:$BACKEND_PORT app:app &
# backend_pid=$!
# echo "Backend server started with PID $backend_pid" | logger

# # Start the frontend server and log its output
# echo "Starting frontend server on port $PORT" | logger
# streamlit run frontend.py --server.port $PORT &
# frontend_pid=$!
# echo "Frontend server started with PID $frontend_pid" | logger

# # Log the end of the script
# echo "start_servers.sh script finished" | logger
