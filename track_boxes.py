#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from collections import defaultdict

def extract_writing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.bitwise_not(inverted)
    return result, contours

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: (x1, y1, w1, h1)
        box2: (x2, y2, w2, h2)
    
    Returns:
        IoU value between 0 and 1
    """
    # Convert to (x1, y1, x2, y2) format
    box1_x1, box1_y1, box1_w, box1_h = box1
    box1_x2, box1_y2 = box1_x1 + box1_w, box1_y1 + box1_h
    
    box2_x1, box2_y1, box2_w, box2_h = box2
    box2_x2, box2_y2 = box2_x1 + box2_w, box2_y1 + box2_h
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def compute_white_to_black_ratio(frame, box):
    """
    Calculate the ratio of white pixels to black pixels within a bounding box
    after processing the frame with extract_writing.
    
    Args:
        frame: Original video frame
        box: (x, y, w, h) of the bounding box
    
    Returns:
        Ratio of white pixels to black pixels
    """
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Extract the region of interest
    roi = frame[y:y+h, x:x+w]
    
    if roi.size == 0:  # Check if ROI is empty
        return 0.0
    
    # Process the ROI
    processed_roi, _ = extract_writing(roi)
    
    # Count white and black pixels
    white_pixels = np.sum(processed_roi == 255)
    black_pixels = np.sum(processed_roi == 0)
    
    # Compute ratio
    ratio = white_pixels / black_pixels if black_pixels > 0 else float('inf')
    return ratio

def track_boxes(bounding_boxes_json, video_path, output_path="tracked_boxes.json", iou_threshold=0.5):
    """
    Track bounding boxes across frames and calculate white-to-black pixel ratios.
    
    Args:
        bounding_boxes_json: Path to JSON file with detected bounding boxes
        video_path: Path to the input video
        output_path: Path to save the tracking results
        iou_threshold: Minimum IoU to consider boxes as the same object
    """
    # Load bounding boxes
    with open(bounding_boxes_json, 'r') as f:
        boxes_data = json.load(f)
    
    # Convert keys to integers (frame numbers)
    boxes_data = {int(k): v for k, v in boxes_data.items()}
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_video.mp4', fourcc, 30.0, (frame_width, frame_height))
    
    # Dictionary to store tracked objects
    tracked_objects = {}
    next_object_id = 0
    
    # Dictionary to store the white-to-black pixel ratio history for each object
    pixel_ratio_history = defaultdict(list)
    
    # Dictionary to maintain object trajectories
    object_trajectories = defaultdict(list)
    
    # Process each frame
    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip if there are no bounding boxes for this frame
        if frame_num not in boxes_data:
            continue
        
        current_boxes = boxes_data[frame_num]
        current_tracked_boxes = {}
        
        # If this is the first frame with boxes, initialize tracked objects
        if not tracked_objects:
            for box in current_boxes:
                tracked_objects[next_object_id] = box
                
                # Calculate white-to-black ratio for this box
                ratio = compute_white_to_black_ratio(frame, box)
                pixel_ratio_history[next_object_id].append((frame_num, ratio))
                
                # Initialize trajectory
                object_trajectories[next_object_id].append((frame_num, box))
                
                next_object_id += 1
        else:
            # Match current boxes to existing tracked objects
            matched_indices = set()
            
            # For each tracked object, find the best matching box
            for obj_id, tracked_box in tracked_objects.items():
                best_iou = 0
                best_box_idx = None
                
                for i, current_box in enumerate(current_boxes):
                    if i in matched_indices:
                        continue
                    
                    iou = calculate_iou(tracked_box, current_box)
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_box_idx = i
                
                if best_box_idx is not None:
                    # Update the tracked object
                    current_tracked_boxes[obj_id] = current_boxes[best_box_idx]
                    matched_indices.add(best_box_idx)
                    
                    # Calculate white-to-black ratio
                    ratio = compute_white_to_black_ratio(frame, current_boxes[best_box_idx])
                    pixel_ratio_history[obj_id].append((frame_num, ratio))
                    
                    # Update trajectory
                    object_trajectories[obj_id].append((frame_num, current_boxes[best_box_idx]))
            
            # Add new objects for unmatched boxes
            for i, box in enumerate(current_boxes):
                if i not in matched_indices:
                    current_tracked_boxes[next_object_id] = box
                    
                    # Calculate white-to-black ratio
                    ratio = compute_white_to_black_ratio(frame, box)
                    pixel_ratio_history[next_object_id].append((frame_num, ratio))
                    
                    # Initialize trajectory
                    object_trajectories[next_object_id].append((frame_num, box))
                    
                    next_object_id += 1
        
        # Update tracked objects for the next frame
        tracked_objects = current_tracked_boxes
        
        # Draw bounding boxes and IDs on the frame for visualization
        frame_vis = frame.copy()
        for obj_id, (x, y, w, h) in current_tracked_boxes.items():
            cv2.rectangle(frame_vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(frame_vis, f"ID: {obj_id}", (int(x), int(y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write the frame to output video
        out.write(frame_vis)
    
    # Release resources
    cap.release()
    out.release()
    
    # Save tracking results
    result = {
        "trajectories": {str(obj_id): traj for obj_id, traj in object_trajectories.items()},
        "pixel_ratios": {str(obj_id): ratios for obj_id, ratios in pixel_ratio_history.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    print(f"Tracking completed. Results saved to {output_path}")
    print(f"Visualization saved to tracked_video.mp4")
    
    return result

def analyze_tracked_boxes(tracking_results):
    """
    Analyze the tracked boxes and pixel ratio history.
    """
    pixel_ratios = tracking_results["pixel_ratios"]
    trajectories = tracking_results["trajectories"]
    
    print(f"Found {len(pixel_ratios)} tracked objects")
    
    # Analyze each object
    for obj_id, ratios in pixel_ratios.items():
        frames, ratio_values = zip(*ratios)
        avg_ratio = np.mean(ratio_values)
        max_ratio = np.max(ratio_values)
        min_ratio = np.min(ratio_values)
        
        trajectory = trajectories[obj_id]
        duration = len(trajectory)
        
        print(f"Object {obj_id}:")
        print(f"  Duration: {duration} frames")
        print(f"  Average white-to-black ratio: {avg_ratio:.2f}")
        print(f"  Min/Max ratio: {min_ratio:.2f}/{max_ratio:.2f}")
        
        # Determine if the object is likely handwritten content
        # (You may want to adjust these thresholds based on your observations)
        if avg_ratio < 5.0 and duration > 10:
            print(f"  Analysis: Likely contains handwritten content")
        else:
            print(f"  Analysis: Likely background or UI element")
        print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python track_boxes.py <bounding_boxes_json> <input_video_path> [output_json_path]")
        return
    
    bounding_boxes_json = sys.argv[1]
    input_video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "tracked_boxes.json"
    
    if not os.path.exists(bounding_boxes_json):
        print(f"Error: Input JSON file '{bounding_boxes_json}' does not exist.")
        return
    
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file '{input_video_path}' does not exist.")
        return
    
    print(f"Processing bounding boxes from {bounding_boxes_json}")
    print(f"Using video from {input_video_path}")
    
    tracking_results = track_boxes(bounding_boxes_json, input_video_path, output_path)
    analyze_tracked_boxes(tracking_results)

if __name__ == "__main__":
    main()
