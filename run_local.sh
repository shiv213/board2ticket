#!/bin/bash

# Run the Whiteboard Video Processing System locally for testing

# Check if a video file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_video_path> [context_file_path]"
    exit 1
fi

VIDEO_PATH=$1
CONTEXT_PATH=${2:-"example_context.json"}

# Create necessary directories
mkdir -p /tmp/whiteboard-processor

# Set up environment
echo "Setting up environment..."
export STORAGE_DIR="/tmp/whiteboard-processor"
export STAGE="local"

# Start the API server in the background
echo "Starting API server..."
cd "$(dirname "$0")"
python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for the API server to start
echo "Waiting for API server to start..."
sleep 5

# Process the video
echo "Processing video: $VIDEO_PATH"
if [ -f "$CONTEXT_PATH" ]; then
    python test_api.py --video "$VIDEO_PATH" --context "$CONTEXT_PATH"
else
    python test_api.py --video "$VIDEO_PATH"
fi

# Clean up
echo "Cleaning up..."
kill $API_PID

echo "Done!"
