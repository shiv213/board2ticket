import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_heat_difference_and_draw_rectangles(frame1, frame2, threshold_value=50):
    """
    Computes the absolute difference between two frames, thresholds the difference, 
    and draws rectangles around significant changes in the frames.
    
    Args:
    - frame1 (np.array): The first frame in BGR format.
    - frame2 (np.array): The second frame in BGR format.
    - threshold_value (int): The threshold for detecting significant changes (default=50).
    
    Returns:
    - None: Displays the results with rectangles around detected changes and a heatmap.
    """
    # Ensure the frames are of the same size
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same dimensions for comparison.")

    # Convert frames to grayscale (if needed)
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the two frames
    diff = cv2.absdiff(gray_frame1, gray_frame2)

    # Threshold the difference to highlight significant changes
    _, thresh_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours from the thresholded difference
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the first frame to draw rectangles on
    frame_with_rectangles = frame1.copy()

    # Draw rectangles around significant contours
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Only draw rectangles around significant contours (you can adjust this by size if needed)
        if w > 30 and h > 30:  # Threshold size to avoid drawing too many small rectangles
            cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Normalize the difference for better visualization (0 to 255)
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Create a heatmap using the difference
    heatmap = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Display the results
    plt.figure(figsize=(20, 10))

    # Frame with rectangles drawn around detected changes
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB))
    plt.title("Detected Changes with Rectangles")
    plt.axis('off')

    # Heatmap of the difference
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap of Frame Differences")
    plt.axis('off')

    plt.show()

def compute_heat_difference_and_merge_rectangles(frame1, frame2, threshold_value=2e4, grid_size=(50, 50), proximity_threshold=1):
    """
    Computes the absolute difference between two frames, detects significant changes 
    in subgrids, and merges adjacent rectangles representing the changes based on proximity.
    
    Args:
    - frame1 (np.array): The first frame in BGR format.
    - frame2 (np.array): The second frame in BGR format.
    - threshold_value (float): The threshold for detecting significant changes (default=2e4).
    - grid_size (tuple): The size of the subgrids (default=(50, 50)).
    - proximity_threshold (int): The threshold for merging rectangles based on proximity (default=1).
    
    Returns:
    - None: Displays the results with merged rectangles around detected changes and a heatmap.
    """
    # Ensure the frames are of the same size
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same dimensions for comparison.")

    # Convert frames to grayscale (if needed)
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the two frames
    diff = cv2.absdiff(gray_frame1, gray_frame2)

    # Create a heatmap using the difference
    heatmap = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)

    # Create a copy of the first frame to draw rectangles on
    frame_with_rectangles = frame1.copy()

    detected_rectangles = []

    # Loop over the image in subgrids
    grid_height, grid_width = grid_size
    height, width = frame1.shape[:2]

    for y in range(0, height, grid_height):
        for x in range(0, width, grid_width):
            # Define the subgrid (region of interest)
            subgrid_diff = diff[y:y + grid_height, x:x + grid_width]

            # Sum the difference in the subgrid to detect significant changes
            change_in_subgrid = np.sum(subgrid_diff)

            # If the change exceeds a threshold, record the rectangle coordinates
            if change_in_subgrid > threshold_value:
                detected_rectangles.append((x, y, x + grid_width, y + grid_height))

    def merge_rectangles(rectangles, proximity_threshold):
        """
        Merges adjacent rectangles based on proximity.

        Args:
        - rectangles (list): List of bounding rectangles (x1, y1, x2, y2).
        - proximity_threshold (int): The maximum distance to consider for merging rectangles.

        Returns:
        - list: List of merged rectangles.
        """
        merged = []
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            found_merge = False
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # Check if the rectangle is close to an existing one (proximity check)
                if abs(x1 - mx2) < proximity_threshold and abs(y1 - my2) < proximity_threshold:
                    # Merge the rectangles by adjusting the coordinates
                    merged[i] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
                    found_merge = True
                    break
            if not found_merge:
                merged.append((x1, y1, x2, y2))
        return merged

    # Merge rectangles that are close to each other
    merged_rectangles = merge_rectangles(detected_rectangles, proximity_threshold)

    # Draw merged rectangles on the frame
    for rect in merged_rectangles:
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame_with_rectangles, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the results
    plt.figure(figsize=(20, 10))

    # Frame with rectangles drawn around detected changes
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB))
    plt.title("Detected Changes with Merged Rectangles")
    plt.axis('off')

    # Heatmap of the difference
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap of Frame Differences")
    plt.axis('off')

    # Difference Image (raw difference)
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title("Raw Difference Image")
    plt.axis('off')

    plt.show()

def difference_of_frames(input_video_path: str):
    """
    Description: This function demonstrates how to compute the difference between two frames of a video,
    detect significant changes, and draw rectangles around the detected changes. It uses OpenCV for video
    capture and processing, and matplotlib for displaying the results.
    
    Returns:
    - None: Displays the results with rectangles around detected changes and a heatmap.
    """

    # Capture video frames
    cap = cv2.VideoCapture(input_video_path)

    # Capture the initial frame (frame 0)
    frame_idx = 0
    init_frame = None
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, init_frame = cap.read()
        if ret:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not read frame at index", frame_idx)

    # Capture the current frame (frame 5000)
    frame_idx = 5000
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not read frame at index", frame_idx)

    compute_heat_difference_and_draw_rectangles(frame.copy(), init_frame.copy())
    compute_heat_difference_and_merge_rectangles(frame.copy(), init_frame.copy())

import cv2
import numpy as np

def extract_writing(frame):
    """
    Extracts black regions that are fully surrounded by white from a whiteboard image.
    Also removes the largest contour (likely a person).
    
    Args:
        frame (np.array): Input frame/image from whiteboard video.

    Returns:
        np.array: Processed image with only black regions surrounded by white and without the person.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(inverted, [largest_contour], -1, 0, thickness=cv2.FILLED)
    
    return inverted

def merge_rectangles(rectangles, proximity_threshold):
    """
    Merges rectangles that are close to each other based on a proximity threshold.
    
    Args:
        rectangles (list of tuples): List of bounding boxes as (x1, y1, x2, y2).
        proximity_threshold (int): Maximum distance to consider two rectangles as mergeable.
    
    Returns:
        list of tuples: Merged bounding boxes.
    """
    merged = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        found_merge = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged):
            if abs(x1 - mx2) < proximity_threshold and abs(y1 - my2) < proximity_threshold:
                merged[i] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
                found_merge = True
                break
        if not found_merge:
            merged.append((x1, y1, x2, y2))
    return merged

def compute_heat_difference_and_merge_rectangles(frame1, threshold_value=2e4, grid_size=(50, 50), proximity_threshold=1):
    """
    Computes the difference between two frames and detects regions of change.
    
    Args:
        frame1 (np.array): The first frame to compare.
        threshold_value (float): Threshold for detecting significant changes.
        grid_size (tuple): Size of the grid for breaking down the image.
        proximity_threshold (int): Maximum distance to merge detected rectangles.
    
    Returns:
        np.array: Frame with detected changes highlighted by bounding boxes.
    """
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = np.zeros_like(frame1)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_frame1, gray_frame2)
    detected_rectangles = []
    grid_height, grid_width = grid_size
    height, width = frame1.shape[:2]
    
    for y in range(0, height, grid_height):
        for x in range(0, width, grid_width):
            subgrid_diff = diff[y:y + grid_height, x:x + grid_width]
            change_in_subgrid = np.sum(subgrid_diff)
            if change_in_subgrid > threshold_value:
                detected_rectangles.append((x, y, x + grid_width, y + grid_height))
    
    merged_rectangles = merge_rectangles(detected_rectangles, proximity_threshold)
    frame_with_rectangles = frame1.copy()
    for rect in merged_rectangles:
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame_with_rectangles, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame_with_rectangles

def process_frames(video_path, frame_nums):
    """
    Processes multiple frames from a video given their indices, applying writing extraction 
    and heat difference detection.
    
    Args:
        video_path (str): Path to the video file.
        frame_nums (list): List of frame indices to process.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error reading frame {frame_num}")
            continue
        
        processed_frame = extract_writing(frame)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        frame_difference = compute_heat_difference_and_merge_rectangles(processed_frame)
        combined = np.hstack((frame, processed_frame, frame_difference))
        
        cv2.imshow("Combined Detection: ", combined)
        cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    input_movie_file = "input_video.mkv"
    frame_indices = list(range(0, 15000, 500))
    process_frames(input_movie_file, frame_indices)

if __name__ == "__main__":
    main()
    