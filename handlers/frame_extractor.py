#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frame Extractor Handler for Whiteboard Video Processing System

This module serves as a serverless function handler for extracting frames from videos.
It downloads the video from S3, extracts frames, and uploads them back to S3.
"""

import os
import json
import boto3
import logging
import tempfile
import cv2
import numpy as np
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Get environment variables
STAGE = os.environ.get('STAGE', 'dev')
STORAGE_BUCKET = os.environ.get('STORAGE_BUCKET', f'{STAGE}-whiteboard-processor-storage')

def extract_frames(video_path: str, sampling_rate: int = 30) -> List[Dict]:
    """
    Extract frames from a video at a specified sampling rate.
    
    Args:
        video_path: Path to the video file
        sampling_rate: Number of frames to skip between extractions
        
    Returns:
        frames: List of extracted frames with metadata
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Extracting frames from video with {total_frames} total frames at {fps} FPS")
    
    # Extract frames at the specified sampling rate
    for i in range(0, total_frames, sampling_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {i}")
            continue
        
        # Save frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            frame_path = temp_file.name
            cv2.imwrite(frame_path, frame)
        
        # Add frame to the list
        frames.append({
            "frame_num": i,
            "timestamp": i / fps,
            "path": frame_path
        })
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    
    return frames

def update_job_status(job_id: str, step: str, status: str, details: Dict = None) -> None:
    """
    Update the status of a job step.
    
    Args:
        job_id: Unique identifier for the job
        step: Name of the step
        status: Status of the step (e.g., "in_progress", "completed", "failed")
        details: Additional details about the step
    """
    try:
        # Get current job status
        status_key = f"status/{job_id}.json"
        response = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=status_key
        )
        
        # Parse job status
        job_status = json.loads(response['Body'].read().decode('utf-8'))
        
        # Update step status
        job_status["steps"][step] = {
            "status": status,
            **(details or {})
        }
        
        # Save updated job status
        s3.put_object(
            Bucket=STORAGE_BUCKET,
            Key=status_key,
            Body=json.dumps(job_status),
            ContentType="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error updating job status: {e}")

def handler(event, context):
    """
    Lambda handler for extracting frames from a video.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        result: Result of the frame extraction
    """
    try:
        # Get job ID and video key from event
        job_id = event["job_id"]
        video_key = event["video_key"]
        
        logger.info(f"Processing job {job_id}, video {video_key}")
        
        # Update job status
        update_job_status(job_id, "frame_extraction", "in_progress")
        
        # Download video from S3
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            video_path = temp_file.name
            s3.download_file(STORAGE_BUCKET, video_key, video_path)
        
        # Extract frames
        frames = extract_frames(video_path)
        
        # Upload frames to S3
        frame_keys = []
        for frame in frames:
            frame_num = frame["frame_num"]
            frame_path = frame["path"]
            frame_key = f"frames/{job_id}/{frame_num}.png"
            
            # Upload frame to S3
            s3.upload_file(frame_path, STORAGE_BUCKET, frame_key)
            
            # Add S3 key to frame metadata
            frame["s3_key"] = frame_key
            
            # Delete temporary file
            os.unlink(frame_path)
            
            # Add to list of frame keys
            frame_keys.append(frame_key)
        
        # Save frame metadata to S3
        frames_key = f"metadata/{job_id}/frames.json"
        s3.put_object(
            Bucket=STORAGE_BUCKET,
            Key=frames_key,
            Body=json.dumps({
                "frames": [
                    {
                        "frame_num": frame["frame_num"],
                        "timestamp": frame["timestamp"],
                        "s3_key": frame["s3_key"]
                    }
                    for frame in frames
                ]
            }),
            ContentType="application/json"
        )
        
        # Update job status
        update_job_status(
            job_id, 
            "frame_extraction", 
            "completed", 
            {
                "frames_count": len(frames),
                "frames_key": frames_key
            }
        )
        
        # Delete temporary video file
        os.unlink(video_path)
        
        # Invoke next function in the pipeline
        lambda_client.invoke(
            FunctionName=f"whiteboard-processor-{STAGE}-extract_audio",
            InvocationType="Event",
            Payload=json.dumps({
                "job_id": job_id,
                "video_key": video_key,
                "frames_key": frames_key,
                "context": event.get("context", {})
            })
        )
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "job_id": job_id,
                "frames_count": len(frames),
                "frames_key": frames_key
            })
        }
        
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        
        # Update job status
        if 'job_id' in locals():
            update_job_status(job_id, "frame_extraction", "failed", {"error": str(e)})
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
