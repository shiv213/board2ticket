#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import sys
import os

def extract_writing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.bitwise_not(inverted)
    return result, contours

def draw_bounding_boxes(frame, contours, min_contour_area=150):
    # Create a copy of the frame for drawing boxes
    bbox_frame = frame.copy()
    if len(bbox_frame.shape) == 2:
        bbox_frame = cv2.cvtColor(bbox_frame, cv2.COLOR_GRAY2BGR)
    
    # List to store bounding box coordinates (x, y, w, h)
    bounding_boxes = []
    
    for contour in contours:
        # Filter out very small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        
        # Draw rectangle
        # cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Use DBSCAN to cluster bounding boxes
    if bounding_boxes:
        clustered_boxes = cluster_boxes_dbscan(bounding_boxes)
        bbox_frame = draw_clustered_boxes(bbox_frame, clustered_boxes)
        return bbox_frame, clustered_boxes
    
    return bbox_frame, bounding_boxes

def cluster_boxes_dbscan(bounding_boxes, eps=100, min_samples=1):
    """
    Cluster bounding boxes using DBSCAN
    
    Args:
        bounding_boxes: List of (x, y, w, h) tuples
        eps: The maximum distance between two samples for them to be considered in the same neighborhood
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        List of merged bounding boxes after clustering
    """
    if not bounding_boxes:
        return []
    
    # Convert boxes to points for clustering (using center points)
    centers = []
    for x, y, w, h in bounding_boxes:
        centers.append([x + w/2, y + h/2])
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # Group boxes by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(bounding_boxes[i])
    
    # Merge boxes in each cluster
    merged_boxes = []
    for label, boxes in clusters.items():
        # Skip noise points (label = -1) if they exist and we want to filter them
        # if label == -1:
        #    continue
        
        # Merge all boxes in this cluster
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[0] + box[2] for box in boxes)
        max_y = max(box[1] + box[3] for box in boxes)
        
        merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    return merged_boxes

def draw_clustered_boxes(frame, clustered_boxes):
    """
    Draw the clustered bounding boxes on a frame
    """
    clustered_frame = frame.copy()
    
    for i, (x, y, w, h) in enumerate(clustered_boxes):

        # If box is larger than 50% of the frame, skip it
        if w > 0.4 * frame.shape[1] or h > 0.5 * frame.shape[0]:
            continue
        else:
            # Draw rectangle with a different color
            cv2.rectangle(clustered_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add box ID text
            cv2.putText(clustered_frame, f"Cluster {i+1}", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 1)
        
    return clustered_frame

def get_largest_contour(contours):
    """
    Find the largest contour from the list of contours
    
    Args:
        contours: List of contours
        
    Returns:
        The largest contour or None if no contours
    """
    if not contours:
        return None
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def box_intersects_contour(box, contour):
    """
    Check if a bounding box intersects with a contour
    
    Args:
        box: Tuple (x, y, w, h)
        contour: Contour points
        
    Returns:
        True if the box intersects with the contour, False otherwise
    """
    if contour is None:
        return False
        
    # Create a mask of the contour
    x, y, w, h = box
    mask = np.zeros((1080, 1920), dtype=np.uint8)  # Assuming standard video dimensions
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Check if any part of the box overlaps with the contour
    box_roi = mask[y:y+h, x:x+w]
    return np.any(box_roi > 0)

def calculate_white_percentage(frame, box):
    """
    Calculate the percentage of white pixels in a box region
    
    Args:
        frame: Binary frame where white represents writing
        box: Tuple (x, y, w, h)
        
    Returns:
        Percentage of white pixels in the box
    """
    x, y, w, h = box
    # Ensure the coordinates are within frame bounds
    max_y, max_x = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, max_x - x)
    h = min(h, max_y - y)
    
    if w <= 0 or h <= 0:
        return 0.0
    
    # Extract the region
    roi = frame[y:y+h, x:x+w]
    
    # Count white pixels (255) and calculate percentage
    white_pixels = cv2.countNonZero(roi)
    total_pixels = w * h
    percentage = (white_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    return percentage

def track_boxes(prev_tracks, current_boxes, max_distance=50):
    """
    Match current boxes with existing tracks
    
    Args:
        prev_tracks: Dictionary of existing tracks
        current_boxes: List of current frame boxes
        max_distance: Maximum distance for matching boxes
        
    Returns:
        Updated tracks dictionary
    """
    if not prev_tracks:
        # First frame, initialize tracks
        tracks = {}
        for i, box in enumerate(current_boxes):
            tracks[i] = {
                'boxes': [box],
                'white_percentages': [],
                'last_seen': 0
            }
        return tracks
    
    # Prepare for matching
    matched_tracks = {}
    matched_indices = set()
    
    # For each existing track, find the closest box
    for track_id, track_info in prev_tracks.items():
        if not track_info['boxes']:
            continue
            
        last_box = track_info['boxes'][-1]
        last_x = last_box[0] + last_box[2] / 2
        last_y = last_box[1] + last_box[3] / 2
        
        best_match_idx = -1
        min_distance = float('inf')
        
        # Find the closest box to this track
        for i, box in enumerate(current_boxes):
            if i in matched_indices:
                continue
                
            curr_x = box[0] + box[2] / 2
            curr_y = box[1] + box[3] / 2
            
            distance = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
            
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                best_match_idx = i
        
        # If found a match
        if best_match_idx >= 0:
            matched_indices.add(best_match_idx)
            matched_box = current_boxes[best_match_idx]
            
            matched_tracks[track_id] = {
                'boxes': prev_tracks[track_id]['boxes'] + [matched_box],
                'white_percentages': prev_tracks[track_id]['white_percentages'].copy(),
                'last_seen': 0
            }
        else:
            # No match found, increment last_seen counter
            matched_tracks[track_id] = {
                'boxes': prev_tracks[track_id]['boxes'],
                'white_percentages': prev_tracks[track_id]['white_percentages'].copy(),
                'last_seen': prev_tracks[track_id]['last_seen'] + 1
            }
    
    # Add new tracks for unmatched boxes
    next_track_id = max(prev_tracks.keys()) + 1 if prev_tracks else 0
    for i, box in enumerate(current_boxes):
        if i not in matched_indices:
            matched_tracks[next_track_id] = {
                'boxes': [box],
                'white_percentages': [],
                'last_seen': 0
            }
            next_track_id += 1
    
    # Remove tracks that haven't been seen for too long
    final_tracks = {k: v for k, v in matched_tracks.items() if v['last_seen'] < 10}
    
    return final_tracks

def process_video_for_tracking(video_path, output_folder="tracking_output"):
    """
    Process video frames, track boxes that don't intersect with the largest contour,
    and calculate white pixel percentage over time
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize tracks dictionary
    tracks = {}
    
    print(f"Processing {total_frames} frames...")
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame to extract writing
        processed_frame, contours = extract_writing(frame)
        
        # Get the largest contour (assuming this is the writer/hand)
        largest_contour = get_largest_contour(contours)
        
        # Get all bounding boxes
        _, all_boxes = draw_bounding_boxes(processed_frame, contours)
        
        # Filter boxes that don't intersect with the largest contour
        filtered_boxes = []
        for box in all_boxes:
            if not box_intersects_contour(box, largest_contour):
                filtered_boxes.append(box)
        
        # Track the filtered boxes
        tracks = track_boxes(tracks, filtered_boxes)
        
        # Calculate white percentage for each track's current box
        for track_id, track_info in tracks.items():
            if track_info['last_seen'] == 0 and track_info['boxes']:  # Only for tracks visible in this frame
                current_box = track_info['boxes'][-1]
                # # save box in processed frame
                # x, y, w, h = current_box
                # crop = processed_frame[y:y+h, x:x+w]
                # cv2.imwrite(f"crop_track_{track_id}_frame_{frame_idx}.png", crop)
                # input()
                white_pct = calculate_white_percentage(processed_frame, current_box)
                track_info['white_percentages'].append(white_pct)
        
        # Visualize tracks on frame every 30 frames
        if frame_idx % 30 == 0:
            vis_frame = frame.copy()
            for track_id, track_info in tracks.items():
                if track_info['last_seen'] == 0:  # Only for tracks visible in this frame
                    box = track_info['boxes'][-1]
                    x, y, w, h = box
                    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add track ID and white pixel percentage
                    latest_pct = track_info['white_percentages'][-1] if track_info['white_percentages'] else 0
                    cv2.putText(vis_frame, f"Track {track_id}: {latest_pct:.1f}%", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
            # cv2.imwrite(f"{output_folder}/tracked_frame_{frame_idx}.jpg", vis_frame)
    
    cap.release()
    
    # Save tracking data
    for track_id, track_info in tracks.items():
        if len(track_info['white_percentages']) > 10:  # Only save tracks with sufficient data
            time_points = [i/fps for i in range(len(track_info['white_percentages']))]
            track_data = np.column_stack((time_points, track_info['white_percentages']))
            np.savetxt(f"{output_folder}/track_{track_id}_data.csv", track_data, 
                       delimiter=",", header="time,white_percentage", comments='')
    
    return tracks

def process_frames(video, frame_nums):
    cap = cv2.VideoCapture(video)

    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_num}")
            continue

        bg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame, contours = extract_writing(frame)
        
        # Draw and cluster bounding boxes in one step
        bbox_frame, final_boxes = draw_bounding_boxes(processed_frame, contours)
        
        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        if len(bg_frame.shape) == 2:
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_GRAY2BGR)

        # Combine frames for visualization
        combined = np.hstack((bg_frame, processed_frame, bbox_frame))
        # cv2.imwrite(f"output/processed_frame_{frame_num}.png", combined)
        
        print(f"Frame {frame_num}: Found {len(contours)} contours, " 
              f"final filtered boxes: {len(final_boxes)}")

    cap.release()


def process_and_save_frame(video, frame_num):
    cap = cv2.VideoCapture(video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        print(f"Error reading frame {frame_num}")
        return

    bg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame, contours = extract_writing(frame)
    
    # Draw and cluster bounding boxes in one step
    bbox_frame, final_boxes = draw_bounding_boxes(processed_frame, contours)
    
    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    if len(bg_frame.shape) == 2:
        bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_GRAY2BGR)

    # Combine frames for visualization
    combined = np.hstack((bg_frame, processed_frame, bbox_frame))
    cv2.imwrite(f"temp/processed_frame_{frame_num}.png", processed_frame)
    
    print(f"Frame {frame_num}: Found {len(contours)} contours, " 
            f"final filtered boxes: {len(final_boxes)}")

    cap.release()


def process_return_frame(frame):
    bg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame, contours = extract_writing(frame)
    
    # Draw and cluster bounding boxes in one step
    bbox_frame, final_boxes = draw_bounding_boxes(processed_frame, contours)
    
    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    if len(bg_frame.shape) == 2:
        bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_GRAY2BGR)

    # Combine frames for visualization
    combined = np.hstack((bg_frame, processed_frame, bbox_frame))
    return combined


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video_path> [output_video_path]")
        return
    
    input_video_path = sys.argv[1]
    output_video_path = None
    if len(sys.argv) >= 3:
        output_video_path = sys.argv[2]

    if not os.path.exists(input_video_path):
        print(f"Error: Input file '{input_video_path}' does not exist.")
        return
        
    # Process video for tracking non-intersecting boxes
    tracks = process_video_for_tracking(input_video_path, "tracking_output")
    print(f"Processed {len(tracks)} unique tracks in the video")
    
    # Original video processing if output path is provided
    if output_video_path:
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920*3, 1080))    
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(0, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 2 == 0:    
                processed_frame = process_return_frame(frame)
                out.write(processed_frame)

        cap.release()
        out.release() 
    

if __name__ == "__main__":
    main()
    # video_path = 'cropped.mp4'
    # process_and_save_frame(video_path, 255*60)

