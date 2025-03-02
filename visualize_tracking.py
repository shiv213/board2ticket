#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

def plot_pixel_ratio_history(tracking_results, output_dir="plots"):
    """
    Generate plots for white-to-black pixel ratio history of each tracked object.
    
    Args:
        tracking_results: Dictionary with tracking results
        output_dir: Directory to save the plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pixel_ratios = tracking_results["pixel_ratios"]
    
    # Create a summary plot with all objects
    plt.figure(figsize=(12, 8))
    
    for obj_id, ratios in pixel_ratios.items():
        frames, ratio_values = zip(*ratios)
        plt.plot(frames, ratio_values, label=f"Object {obj_id}")
    
    plt.xlabel("Frame Number")
    plt.ylabel("White-to-Black Pixel Ratio")
    plt.title("Pixel Ratio History for All Tracked Objects")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "all_objects_ratio_history.png"))
    
    # Create individual plots for each object
    for obj_id, ratios in pixel_ratios.items():
        frames, ratio_values = zip(*ratios)
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, ratio_values, 'b-', linewidth=2)
        plt.xlabel("Frame Number")
        plt.ylabel("White-to-Black Pixel Ratio")
        plt.title(f"Pixel Ratio History for Object {obj_id}")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"object_{obj_id}_ratio_history.png"))
        plt.close()
    
    print(f"Plots saved to {output_dir}")

def create_tracking_visualization(tracking_results, video_path, output_path="tracking_visualization.mp4"):
    """
    Create a video visualization of the tracked objects.
    
    Args:
        tracking_results: Dictionary with tracking results
        video_path: Path to the input video
        output_path: Path to save the visualization video
    """
    # Load trajectories
    trajectories = tracking_results["trajectories"]
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
    
    # Create a dictionary mapping frame numbers to active objects
    frame_to_objects = {}
    for obj_id, traj in trajectories.items():
        for frame_num, box in traj:
            if frame_num not in frame_to_objects:
                frame_to_objects[frame_num] = []
            frame_to_objects[frame_num].append((obj_id, box))
    
    # Process each frame
    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes for active objects in this frame
        if frame_num in frame_to_objects:
            for obj_id, box in frame_to_objects[frame_num]:
                x, y, w, h = [int(val) for val in box]
                
                # Generate a consistent color for each object ID
                color_hash = int(obj_id) * 50 % 255
                color = (color_hash, 255 - color_hash, (color_hash + 125) % 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"ID: {obj_id}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trajectory trail
                obj_trajectory = trajectories[obj_id]
                trail_points = [(int(box[0] + box[2]/2), int(box[1] + box[3]/2)) 
                              for f, box in obj_trajectory if f <= frame_num]
                
                if len(trail_points) > 1:
                    for i in range(1, len(trail_points)):
                        cv2.line(frame, trail_points[i-1], trail_points[i], color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to output
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Visualization saved to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_tracking.py <tracking_results_json> <input_video_path> [output_video_path]")
        return
    
    tracking_results_json = sys.argv[1]
    input_video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "tracking_visualization.mp4"
    
    if not os.path.exists(tracking_results_json):
        print(f"Error: Input JSON file '{tracking_results_json}' does not exist.")
        return
    
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file '{input_video_path}' does not exist.")
        return
    
    # Load tracking results
    with open(tracking_results_json, 'r') as f:
        tracking_results = json.load(f)
    
    # Generate plots
    plot_pixel_ratio_history(tracking_results)
    
    # Create visualization video
    create_tracking_visualization(tracking_results, input_video_path, output_path)

if __name__ == "__main__":
    main()
