#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import sys
import os
import json

def extract_writing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.bitwise_not(inverted)
    return result, contours

def draw_bounding_boxes(frame, contours, min_contour_area=50):
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
        bbox_frame, boxes = draw_clustered_boxes(bbox_frame, clustered_boxes)
        return bbox_frame, boxes
    
    return bbox_frame, clustered_boxes

def cluster_boxes_dbscan(bounding_boxes, eps=100, min_samples=3):
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
    boxes = []
    for i, (x, y, w, h) in enumerate(clustered_boxes):

        # If box is larger than 50% of the frame, skip it
        if w > 0.4 * frame.shape[1] or h > 0.5 * frame.shape[0]:
            continue
        else:
            boxes.append((x, y, w, h))
            # Draw rectangle with a different color
            cv2.rectangle(clustered_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add box ID text
            cv2.putText(clustered_frame, f"Cluster {i+1}", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 1)
        
    return clustered_frame, boxes

def draw_merged_boxes(frame, merged_boxes):
    """
    Draw merged bounding boxes on a frame
    """
    merged_frame = frame.copy()
    if len(merged_frame.shape) == 2:
        merged_frame = cv2.cvtColor(merged_frame, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(merged_boxes):
        # Draw rectangle
        cv2.rectangle(merged_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Add box ID text
        cv2.putText(merged_frame, f"Box {i+1}", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), 1)
    
    return merged_frame

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
        cv2.imwrite(f"output/processed_frame_{frame_num}.png", combined)
        
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
    cv2.imwrite(f"processed_frame_{frame_num}.png", combined)
    
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
    return processed_frame, final_boxes


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
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bounding_box_list = {}

    # out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920*3, 1080))    
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(0, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, bounding_boxes = process_return_frame(frame)
        for j, box in enumerate(bounding_boxes):
            x, y, w, h = box
            # saved cropped bounding box into list
            bounding_box_list[i] = processed_frame[y:y+h, x:x+w]
            
            cv2.imwrite(f"{output_dir}/frame_{i}_{x},{y}_{y+h},{x+w}.png", bounding_box_list[i])
            # input()
        # out.write(processed_frame)
    cap.release()
    # out.release() 
    
if __name__ == "__main__":
    main()
    # video_path = 'cropped.mp4'
    # cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()
    # process_frames(video_path, [i for i in tqdm(range(0, total_frames, 100))])
    # process_and_save_frame(video_path, 255*60)

