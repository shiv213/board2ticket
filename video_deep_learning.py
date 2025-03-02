import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Performs Non-Maximum Suppression (NMS) to remove redundant bounding boxes
    based on their overlap threshold.
    
    Args:
    - boxes (np.array): Array of bounding boxes (x1, y1, x2, y2).
    - probs (list): List of probabilities for each bounding box.
    - overlapThresh (float): Threshold for considering overlap (default=0.3).

    Returns:
    - np.array: The filtered bounding boxes after applying NMS.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs if probs is not None else y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def load_east_detector():
    """
    Loads the pre-trained EAST (Efficient and Accurate Scene Text Detector) model.
    
    Returns:
    - cv2.dnn_Net: The EAST model loaded using OpenCV DNN module.
    """
    print("[INFO] loading EAST text detector...")
    return cv2.dnn.readNet('frozen_east_text_detection.pb')

def get_text_bounding_boxes(image, net):
    """
    Detects bounding boxes around text regions using the EAST text detector.
    
    Args:
    - image (np.array): The input image.
    - net (cv2.dnn_Net): The pre-trained EAST model.
    
    Returns:
    - boxes (np.array): Bounding boxes of detected text regions.
    """
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image_resized = cv2.resize(image, (newW, newH))

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < 0.75:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return non_max_suppression(np.array(rects), probs=confidences)

def detect_content_with_clip(image_frame):
    """
    Uses CLIP model to detect the content of an image and draws bounding boxes
    based on content similarity to predefined labels.

    Args:
    - image_frame (np.array): The input image.
    """
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=["content", "text", "drawing", "diagram"], images=pil_image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    similarities = torch.cosine_similarity(image_features, text_features)
    best_match_idx = similarities.argmax().item()

    if best_match_idx == 0:  # If "content" is detected
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Minimum size threshold
                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
image = cv2.imread('image.jpg')  # Read your image
frame = image.copy()

# Load EAST detector
east_net = load_east_detector()

# Get bounding boxes using EAST detector
boxes = get_text_bounding_boxes(frame, east_net)

# Draw the detected bounding boxes on the image
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Show the processed image with bounding boxes
cv2.imshow("Text Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use CLIP to detect content and draw bounding boxes around text
detect_content_with_clip(frame)
