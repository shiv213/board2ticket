#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Script for Whiteboard Video Processing API

This script demonstrates how to use the Whiteboard Video Processing API
to process a video, check the status of a job, and retrieve the results.
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import Dict, Any, Optional

def process_video(api_url: str, video_path: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Upload a video for processing.
    
    Args:
        api_url: Base URL of the API
        video_path: Path to the video file
        context: Optional context data for processing
        
    Returns:
        job_id: Unique identifier for the processing job
    """
    # Ensure the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Prepare the request
    url = f"{api_url}/process"
    files = {"file": open(video_path, "rb")}
    data = {}
    
    # Add context if provided
    if context:
        data["context"] = json.dumps(context)
    
    # Send the request
    print(f"Uploading video: {video_path}")
    response = requests.post(url, files=files, data=data)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error processing video: {response.text}")
    
    # Parse the response
    result = response.json()
    job_id = result["job_id"]
    
    print(f"Video uploaded successfully. Job ID: {job_id}")
    return job_id

def get_job_status(api_url: str, job_id: str) -> Dict[str, Any]:
    """
    Get the status of a processing job.
    
    Args:
        api_url: Base URL of the API
        job_id: Unique identifier for the processing job
        
    Returns:
        status: Status information for the job
    """
    # Prepare the request
    url = f"{api_url}/status/{job_id}"
    
    # Send the request
    response = requests.get(url)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error getting job status: {response.text}")
    
    # Parse the response
    status = response.json()
    
    return status

def get_job_result(api_url: str, job_id: str) -> Dict[str, Any]:
    """
    Get the result of a completed processing job.
    
    Args:
        api_url: Base URL of the API
        job_id: Unique identifier for the processing job
        
    Returns:
        result: Result data for the job
    """
    # Prepare the request
    url = f"{api_url}/result/{job_id}"
    
    # Send the request
    response = requests.get(url)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error getting job result: {response.text}")
    
    # Parse the response
    result = response.json()
    
    return result

def wait_for_completion(api_url: str, job_id: str, poll_interval: int = 10, timeout: int = 3600) -> Dict[str, Any]:
    """
    Wait for a processing job to complete.
    
    Args:
        api_url: Base URL of the API
        job_id: Unique identifier for the processing job
        poll_interval: Interval between status checks in seconds
        timeout: Maximum time to wait in seconds
        
    Returns:
        result: Result data for the job
    """
    start_time = time.time()
    
    while True:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job did not complete within {timeout} seconds")
        
        # Get the job status
        status = get_job_status(api_url, job_id)
        
        # Print the current status
        print(f"Job status: {status['status']}")
        
        # If the job is completed, get the result
        if status["status"] == "completed":
            return get_job_result(api_url, job_id)
        
        # If the job failed, raise an exception
        if status["status"] == "failed":
            raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
        
        # Wait before checking again
        print(f"Waiting {poll_interval} seconds...")
        time.sleep(poll_interval)

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the Whiteboard Video Processing API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--context", help="Path to a JSON file containing context data")
    parser.add_argument("--job-id", help="Job ID to check (if not uploading a new video)")
    parser.add_argument("--poll-interval", type=int, default=10, help="Interval between status checks in seconds")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum time to wait in seconds")
    
    args = parser.parse_args()
    
    try:
        # Load context data if provided
        context = None
        if args.context:
            with open(args.context, "r") as f:
                context = json.load(f)
        
        # If a job ID is provided, check its status
        if args.job_id:
            job_id = args.job_id
            print(f"Checking status of job: {job_id}")
        else:
            # Otherwise, upload a new video
            job_id = process_video(args.api_url, args.video, context)
        
        # Wait for the job to complete
        result = wait_for_completion(args.api_url, job_id, args.poll_interval, args.timeout)
        
        # Print the result
        print("\nJob completed successfully!")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
