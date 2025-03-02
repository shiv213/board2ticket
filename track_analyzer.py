#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class BoundingBox:
    def __init__(self, frame_num, x1, y1, x2, y2, image=None):
        self.frame_num = frame_num
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.image = image
        self.width = x2 - x1
        self.height = y2 - y1
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.white_pixels = self._count_white_pixels() if image is not None else 0
    
    def _count_white_pixels(self):
        """Count white pixels in the image"""
        if self.image is None:
            return 0
        
        # Ensure image is grayscale
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        # Count non-zero (white) pixels
        return cv2.countNonZero(gray)
    
    def iou(self, other):
        """Calculate Intersection over Union with another bounding box"""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        self_area = self.width * self.height
        other_area = other.width * other.height
        
        return intersection_area / float(self_area + other_area - intersection_area)
    
    def distance(self, other):
        """Calculate Euclidean distance between centers"""
        return np.sqrt((self.center_x - other.center_x)**2 + (self.center_y - other.center_y)**2)

class Track:
    def __init__(self, track_id):
        self.track_id = track_id
        self.boxes = {}  # frame_num -> BoundingBox
        self.last_frame = -1
        self.active = True
    
    def add_box(self, box):
        self.boxes[box.frame_num] = box
        self.last_frame = max(self.last_frame, box.frame_num)
    
    def get_white_pixel_timeline(self):
        """Get white pixel count over time"""
        frames = sorted(self.boxes.keys())
        white_pixels = [self.boxes[frame].white_pixels for frame in frames]
        return frames, white_pixels

def parse_filename(filename):
    """Parse frame number and bounding box coordinates from filename"""
    # Expected format: frame_123_10,20_50,60.png (frame_num_x1,y1_x2,y2.png)
    match = re.match(r'frame_(\d+)_(\d+),(\d+)_(\d+),(\d+)', filename)
    if match:
        frame_num = int(match.group(1))
        x1 = int(match.group(2))
        y1 = int(match.group(3))
        x2 = int(match.group(4))
        y2 = int(match.group(5))
        return frame_num, x1, y1, x2, y2
    else:
        return None

def load_bounding_boxes(output_folder):
    """Load bounding box images from the output folder"""
    boxes_by_frame = defaultdict(list)
    
    for filename in os.listdir(output_folder):
        if filename.endswith('.png') and filename.startswith('frame_'):
            parsed = parse_filename(filename)
            if parsed:
                frame_num, x1, y1, x2, y2 = parsed
                image_path = os.path.join(output_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                box = BoundingBox(frame_num, x1, y1, x2, y2, image)
                boxes_by_frame[frame_num].append(box)
    
    return boxes_by_frame

def create_tracks(boxes_by_frame, max_distance=100, iou_threshold=0.2, max_frames_to_skip=5):
    """Create tracks from bounding boxes across frames"""
    frames = sorted(boxes_by_frame.keys())
    tracks = []
    next_track_id = 0
    active_tracks = []
    
    for frame_idx, frame_num in enumerate(tqdm(frames, desc="Creating tracks")):
        # Get current frame's boxes
        current_boxes = boxes_by_frame[frame_num]
        
        # If it's the first frame, create new tracks for all boxes
        if frame_idx == 0 or not active_tracks:
            for box in current_boxes:
                track = Track(next_track_id)
                track.add_box(box)
                tracks.append(track)
                active_tracks.append(track)
                next_track_id += 1
            continue
        
        # For all other frames, match boxes to existing tracks
        if not current_boxes:
            # No boxes in this frame, just update active status
            for track in active_tracks:
                if frame_num - track.last_frame > max_frames_to_skip:
                    track.active = False
            active_tracks = [t for t in active_tracks if t.active]
            continue
        
        # Calculate cost matrix based on distance and IoU
        cost_matrix = np.zeros((len(active_tracks), len(current_boxes)))
        
        for i, track in enumerate(active_tracks):
            last_box = track.boxes[track.last_frame]
            for j, box in enumerate(current_boxes):
                # Calculate IoU-based cost
                iou = last_box.iou(box)
                dist = last_box.distance(box)
                
                # High cost for low IoU or large distance
                if iou > iou_threshold and dist < max_distance:
                    cost_matrix[i, j] = dist * (1 - iou)
                else:
                    cost_matrix[i, j] = 1000000  # Very high cost for unlikely matches
        
        # Use Hungarian algorithm to find optimal assignment
        track_indices, box_indices = linear_sum_assignment(cost_matrix)
        
        # Mark all boxes as unassigned initially
        assigned_boxes = set()
        
        # Update tracks with new boxes
        for track_idx, box_idx in zip(track_indices, box_indices):
            if cost_matrix[track_idx, box_idx] < 1000000:  # Only assign if cost is reasonable
                track = active_tracks[track_idx]
                box = current_boxes[box_idx]
                track.add_box(box)
                assigned_boxes.add(box_idx)
        
        # Create new tracks for unassigned boxes
        for j, box in enumerate(current_boxes):
            if j not in assigned_boxes:
                track = Track(next_track_id)
                track.add_box(box)
                tracks.append(track)
                active_tracks.append(track)
                next_track_id += 1
        
        # Update active status
        for track in active_tracks:
            if frame_num - track.last_frame > max_frames_to_skip:
                track.active = False
        active_tracks = [t for t in active_tracks if t.active]
    
    return tracks

def visualize_tracks(tracks, video_width=1920, video_height=1080, save_path=None):
    """Visualize tracks and their white pixel counts"""
    min_frames_threshold = 5  # Minimum frames to consider a valid track
    
    # Filter tracks with enough frames
    valid_tracks = [track for track in tracks if len(track.boxes) >= min_frames_threshold]
    
    print(f"Found {len(valid_tracks)} valid tracks out of {len(tracks)} total tracks")
    
    # Create a colormap for visualization
    colors = plt.cm.jet(np.linspace(0, 1, len(valid_tracks)))
    
    plt.figure(figsize=(15, 10))
    
    # Plot each track's white pixel count over time
    for i, track in enumerate(valid_tracks):
        frames, white_pixels = track.get_white_pixel_timeline()
        plt.plot(frames, white_pixels, '-o', color=colors[i], 
                 alpha=0.7, linewidth=2, label=f"Track {track.track_id}")
    
    plt.title("White Pixel Count Over Time for Each Track")
    plt.xlabel("Frame Number")
    plt.ylabel("White Pixel Count")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return valid_tracks

def save_track_data(tracks, output_file):
    """Save track data as JSON"""
    track_data = []
    
    for track in tracks:
        track_info = {
            "track_id": track.track_id,
            "frames": [],
            "white_pixels": [],
            "bounding_boxes": []
        }
        
        # Sort frames chronologically
        frames = sorted(track.boxes.keys())
        
        for frame_num in frames:
            box = track.boxes[frame_num]
            track_info["frames"].append(frame_num)
            track_info["white_pixels"].append(box.white_pixels)
            track_info["bounding_boxes"].append({
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
            })
        
        track_data.append(track_info)
    
    with open(output_file, 'w') as f:
        json.dump(track_data, f, indent=2)
    
    print(f"Saved track data to {output_file}")

def main():
    output_folder = "output"
    if not os.path.exists(output_folder):
        print(f"Error: Output folder '{output_folder}' does not exist.")
        return
    
    # Create results directory if it doesn't exist
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    print("Loading bounding boxes from images...")
    boxes_by_frame = load_bounding_boxes(output_folder)
    print(f"Loaded bounding boxes from {len(boxes_by_frame)} frames")
    
    print("Creating tracks...")
    tracks = create_tracks(boxes_by_frame)
    print(f"Created {len(tracks)} tracks")
    
    print("Visualizing tracks...")
    valid_tracks = visualize_tracks(
        tracks, 
        save_path=os.path.join(results_folder, "track_visualization.png")
    )
    
    print("Saving track data...")
    save_track_data(valid_tracks, os.path.join(results_folder, "track_data.json"))
    
    print("Done!")

if __name__ == "__main__":
    main()
