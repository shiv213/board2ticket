#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Handler for Whiteboard Video Processing System

This module serves as the API handler for the whiteboard video processing system
when deployed as serverless functions. It uses FastAPI to handle API requests and
routes them to the appropriate functions.
"""

import os
import json
import boto3
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Whiteboard Video Processing API", 
              description="API for processing whiteboard videos and extracting content",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS clients
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')
step_functions = boto3.client('stepfunctions')

# Get environment variables
STAGE = os.environ.get('STAGE', 'dev')
STORAGE_BUCKET = os.environ.get('STORAGE_BUCKET', f'{STAGE}-whiteboard-processor-storage')
STATE_MACHINE_ARN = os.environ.get('STATE_MACHINE_ARN', '')

class ContextData(BaseModel):
    """Model for context data provided with video processing requests."""
    codebase_context: Optional[str] = None
    project_name: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    context: Optional[ContextData] = None
):
    """
    Process a whiteboard video.
    
    - **file**: The video file to process
    - **context**: Optional context data for processing
    
    Returns:
        job_id: Unique identifier for the processing job
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save file to S3
        video_key = f"videos/{job_id}/{file.filename}"
        file_content = await file.read()
        s3.put_object(
            Bucket=STORAGE_BUCKET,
            Key=video_key,
            Body=file_content
        )
        
        # Convert context to dict if provided
        context_dict = context.dict() if context else {}
        
        # Create job status object
        job_status = {
            "job_id": job_id,
            "status": "started",
            "video_key": video_key,
            "context": context_dict,
            "steps": {}
        }
        
        # Save job status to S3
        status_key = f"status/{job_id}.json"
        s3.put_object(
            Bucket=STORAGE_BUCKET,
            Key=status_key,
            Body=json.dumps(job_status),
            ContentType="application/json"
        )
        
        # Start Step Functions execution
        if STATE_MACHINE_ARN:
            step_functions.start_execution(
                stateMachineArn=STATE_MACHINE_ARN,
                name=job_id,
                input=json.dumps({
                    "job_id": job_id,
                    "video_key": video_key,
                    "context": context_dict
                })
            )
        else:
            # For testing without Step Functions
            # Invoke the first Lambda function directly
            lambda_client.invoke(
                FunctionName=f"whiteboard-processor-{STAGE}-extract_frames",
                InvocationType="Event",
                Payload=json.dumps({
                    "job_id": job_id,
                    "video_key": video_key,
                    "context": context_dict
                })
            )
        
        return {"job_id": job_id, "status": "processing"}
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get the status of a processing job.
    
    - **job_id**: Unique identifier for the processing job
    
    Returns:
        status: Status information for the job
    """
    try:
        # Get job status from S3
        status_key = f"status/{job_id}.json"
        response = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=status_key
        )
        
        # Parse job status
        job_status = json.loads(response['Body'].read().decode('utf-8'))
        
        return job_status
    
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Job not found")
    
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Get the result of a completed processing job.
    
    - **job_id**: Unique identifier for the processing job
    
    Returns:
        result: Result data for the job
    """
    try:
        # Get job status from S3
        status_key = f"status/{job_id}.json"
        response = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=status_key
        )
        
        # Parse job status
        job_status = json.loads(response['Body'].read().decode('utf-8'))
        
        # Check if job is completed
        if job_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")
        
        # Get result from S3
        result_key = f"results/{job_id}.json"
        response = s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=result_key
        )
        
        # Parse result
        result = json.loads(response['Body'].read().decode('utf-8'))
        
        return result
    
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Job or result not found")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting job result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Create handler for AWS Lambda
handler = Mangum(app)
