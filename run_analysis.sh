#!/bin/bash

# Check if a video path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_video_path>"
    exit 1
fi

VIDEO_PATH=$1

# Create necessary directories
mkdir -p output results

# Step 1: Process video and extract bounding boxes
echo "Processing video to extract bounding boxes..."
python main.py "$VIDEO_PATH"

# Step 2: Analyze output and create tracks
echo "Creating and analyzing tracks from bounding boxes..."
python track_analyzer.py

# Step 3: View results
echo "Results saved to results/track_data.json and results/track_visualization.png"
echo ""
echo "To view interactive track animation:"
echo "python track_viewer.py --video $VIDEO_PATH"
echo ""
echo "To view white pixel timeline:"
echo "python track_viewer.py --mode timeline"

# Make the script executable
chmod +x run_analysis.sh
