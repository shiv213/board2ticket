#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend when saving video

def load_track_data(json_file):
    """Load track data from JSON file"""
    with open(json_file, 'r') as f:
        track_data = json.load(f)
    return track_data

def visualize_track_timeline(track_data):
    """Visualize white pixel count timeline for all tracks"""
    plt.figure(figsize=(15, 8))
    
    for track in track_data:
        track_id = track["track_id"]
        frames = track["frames"]
        white_pixels = track["white_pixels"]
        
        plt.plot(frames, white_pixels, '-o', label=f"Track {track_id}", alpha=0.7)
    
    plt.title("White Pixel Count Over Time for Each Track")
    plt.xlabel("Frame Number")
    plt.ylabel("White Pixel Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_track_animation(track_data, video_path=None, video_width=1920, video_height=1080):
    """Create an interactive animation showing tracks on video frames"""
    # If a video path is provided, we'll display tracks on top of the video frames
    video_available = False
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        video_available = True
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Find max frame number
    all_frames = []
    for track in track_data:
        all_frames.extend(track["frames"])
    min_frame = min(all_frames) if all_frames else 0
    max_frame = max(all_frames) if all_frames else 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Create frame slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=min_frame,
        valmax=max_frame,
        valinit=min_frame,
        valstep=1
    )
    
    # Function to update plot based on slider
    def update(val):
        frame_num = int(frame_slider.val)
        ax.clear()
        
        # If video is available, show the frame
        if video_available:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame)
            else:
                ax.set_xlim(0, video_width)
                ax.set_ylim(video_height, 0)  # Invert y-axis to match image coordinates
        else:
            # If no video, just show an empty canvas
            ax.set_xlim(0, video_width)
            ax.set_ylim(video_height, 0)  # Invert y-axis to match image coordinates
        
        # Draw bounding boxes for each track at current frame
        for track in track_data:
            track_id = track["track_id"]
            frames = track["frames"]
            
            if frame_num in frames:
                idx = frames.index(frame_num)
                bbox = track["bounding_boxes"][idx]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                
                # Add track ID text
                ax.text(x1, y1-5, f"Track {track_id}", color='r', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
                
                # Add white pixel count
                white_pixels = track["white_pixels"][idx]
                ax.text(x1, y1+15, f"Pixels: {white_pixels}", color='blue', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f"Frame {frame_num}")
        fig.canvas.draw_idle()
    
    # Connect the update function to the slider
    frame_slider.on_changed(update)
    
    # Initialize with first frame
    update(min_frame)
    
    plt.show()
    
    # Clean up video capture if used
    if cap is not None:
        cap.release()

def save_track_animation_as_video(track_data, output_path, video_path=None, fps=30, video_width=1920, video_height=1080):
    """Save track animation as a video file"""
    # If a video path is provided, we'll display tracks on top of the video frames
    video_available = False
    input_cap = None
    if video_path and os.path.exists(video_path):
        input_cap = cv2.VideoCapture(video_path)
        video_available = True
        video_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Find max frame number
    all_frames = []
    for track in track_data:
        all_frames.extend(track["frames"])
    min_frame = min(all_frames) if all_frames else 0
    max_frame = max(all_frames) if all_frames else 100
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    print(f"Saving animation to {output_path}...")
    print(f"Processing frames {min_frame} to {max_frame}")
    
    # Process each frame
    for frame_num in range(min_frame, max_frame + 1):
        # Create a figure for this frame
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.tight_layout()
        
        # If video is available, show the frame
        if video_available:
            input_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = input_cap.read()
            if ret:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
            else:
                ax.set_xlim(0, video_width)
                ax.set_ylim(video_height, 0)  # Invert y-axis to match image coordinates
        else:
            # If no video, just show an empty canvas
            ax.set_xlim(0, video_width)
            ax.set_ylim(video_height, 0)  # Invert y-axis to match image coordinates
            # Set white background
            ax.set_facecolor('white')
        
        # Draw bounding boxes for each track at current frame
        for track in track_data:
            track_id = track["track_id"]
            frames = track["frames"]
            
            if frame_num in frames:
                idx = frames.index(frame_num)
                bbox = track["bounding_boxes"][idx]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                
                # Add track ID text
                ax.text(x1, y1-5, f"Track {track_id}", color='r', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
                
                # Add white pixel count
                white_pixels = track["white_pixels"][idx]
                ax.text(x1, y1+15, f"Pixels: {white_pixels}", color='blue', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f"Frame {frame_num}")
        
        # Remove axes
        ax.set_axis_off()
        
        # Convert matplotlib figure to an OpenCV image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, (video_width, video_height))
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write the frame to video
        out.write(img)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Show progress
        if frame_num % 10 == 0:
            print(f"Processed frame {frame_num}/{max_frame}")
    
    # Release resources
    out.release()
    if input_cap is not None:
        input_cap.release()
    
    print(f"Video saved successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Track Viewer")
    parser.add_argument("--data", default="results/track_data.json", 
                        help="Path to track data JSON file")
    parser.add_argument("--video", default=None, 
                        help="Path to video file (optional)")
    parser.add_argument("--mode", default="animation", choices=["animation", "timeline"],
                        help="Visualization mode: 'animation' or 'timeline'")
    parser.add_argument("--save", default=None,
                        help="Save animation as video file (specify output path)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output video (default: 30)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Track data file '{args.data}' does not exist.")
        return
    
    track_data = load_track_data(args.data)
    
    if args.mode == "timeline":
        visualize_track_timeline(track_data)
    else:
        if args.save:
            save_track_animation_as_video(track_data, args.save, args.video, args.fps)
        else:
            create_track_animation(track_data, args.video)

if __name__ == "__main__":
    main()
