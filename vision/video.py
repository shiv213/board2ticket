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

def process_bounding_boxes(frame, contours, min_contour_area=50, cluster=True, eps=100, min_samples=3, 
                           filter_large_boxes=True, draw_boxes=True, box_color=(255, 0, 0)):
    """
    Unified function to process, cluster and draw bounding boxes
    
    Args:
        frame: Input frame
        contours: List of contours
        min_contour_area: Minimum area for considering a contour
        cluster: Whether to cluster boxes using DBSCAN
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        filter_large_boxes: Whether to filter out boxes larger than 40-50% of the frame
        draw_boxes: Whether to draw boxes on the frame
        box_color: Color for drawing boxes (B,G,R)
        
    Returns:
        Processed frame with boxes drawn, list of final bounding boxes
    """
    # Create a copy of the frame for drawing boxes
    bbox_frame = frame.copy()
    if len(bbox_frame.shape) == 2:
        bbox_frame = cv2.cvtColor(bbox_frame, cv2.COLOR_GRAY2BGR)
    
    # List to store bounding box coordinates (x, y, w, h)
    bounding_boxes = []
    
    # Extract individual bounding boxes from contours
    for contour in contours:
        # Filter out very small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bbox_contents = bbox_frame[y:y+h, x:x+w]
        # check if border of bounding box is 70 percent white
        border_thickness = 2
        top = bbox_contents[:border_thickness, :]
        bottom = bbox_contents[-border_thickness:, :]
        left = bbox_contents[:, :border_thickness]
        right = bbox_contents[:, -border_thickness:]
        border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
        white_ratio = np.count_nonzero(border_pixels > 200) / border_pixels.size
        if white_ratio >= 0.7:
            # print(f"White ratio: {white_ratio}")
            bounding_boxes.append((x, y, w, h))
    
    final_boxes = []
    
    # Apply clustering if requested and if we have boxes
    if cluster and bounding_boxes:
        clustered_boxes = cluster_boxes_dbscan(bounding_boxes, eps, min_samples)
        
        # Filter and process the clustered boxes
        for i, (x, y, w, h) in enumerate(clustered_boxes):
            # Filter out large boxes if requested
            if filter_large_boxes and (w > 0.4 * frame.shape[1] or h > 0.5 * frame.shape[0]):
                continue
            
            final_boxes.append((x, y, w, h))
            
            # Draw boxes if requested
            if draw_boxes:
                # Draw rectangle
                cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), box_color, 2)
                
                # Add box ID text
                cv2.putText(bbox_frame, f"Cluster {i+1}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, box_color, 1)
    # If not clustering, use original boxes
    else:
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if filter_large_boxes and (w > 0.4 * frame.shape[1] or h > 0.5 * frame.shape[0]):
                continue
                
            final_boxes.append((x, y, w, h))
            
            # Draw boxes if requested
            if draw_boxes:
                cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(bbox_frame, f"Box {i+1}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, box_color, 1)
    
    return bbox_frame, final_boxes

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
        
        # Use the new unified bounding box function
        bbox_frame, final_boxes = process_bounding_boxes(processed_frame, contours)
        
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
    
    # Use the new unified bounding box function
    bbox_frame, final_boxes = process_bounding_boxes(processed_frame, contours)
    
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
    
    # out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920*3, 1080))    
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(0, total_frames), ascii=True):
        ret, frame = cap.read()
        if not ret:
            break
        inverted_frame, contours = extract_writing(frame)
        bbox_frame, final_boxes = process_bounding_boxes(inverted_frame, contours, draw_boxes=False)
        
        for j, box in enumerate(final_boxes):
            x, y, w, h = box
            # saved cropped bounding box to image
            cv2.imwrite(f"{output_dir}/frame_{i}_{x},{y}_{y+h},{x+w}.png", bbox_frame[y:y+h, x:x+w])
            # input()
        # out.write(inverted_frame)
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

