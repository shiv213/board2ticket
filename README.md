# Whiteboard Video Processing System

A scalable, serverless system for processing whiteboard videos, extracting content, and generating structured outputs.

## Architecture

This system is designed to process whiteboard videos in a scalable, serverless manner. It extracts frames from videos, detects content regions, tracks content across frames, transcribes audio, and generates structured outputs using LLMs.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Extract   │     │   Extract   │     │   Detect    │     │    Track    │
│   Frames    │────▶│    Audio    │────▶│   Content   │────▶│   Content   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Transcribe │     │   Analyze   │     │  Generate   │     │     API     │
│    Audio    │────▶│   Content   │────▶│   Output    │◀────│   Gateway   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Components

1. **API Gateway**: Handles HTTP requests and routes them to the appropriate Lambda functions.
2. **Extract Frames**: Extracts frames from the video at a specified sampling rate.
3. **Extract Audio**: Extracts audio from the video.
4. **Detect Content**: Detects content regions in the extracted frames.
5. **Track Content**: Tracks content across frames to identify changes over time.
6. **Transcribe Audio**: Transcribes audio to text using speech recognition.
7. **Analyze Content**: Analyzes content with transcript and context.
8. **Generate Output**: Generates structured outputs using LLMs.

### Data Flow

1. User uploads a video through the API.
2. The video is stored in S3.
3. A Step Functions workflow is triggered to orchestrate the processing pipeline.
4. Each step in the pipeline is executed as a separate Lambda function.
5. The results are stored in S3 and can be retrieved through the API.

## Deployment

### Prerequisites

- [Node.js](https://nodejs.org/) (v14 or later)
- [Serverless Framework](https://www.serverless.com/) (v3 or later)
- [AWS CLI](https://aws.amazon.com/cli/) (configured with appropriate credentials)
- [Python](https://www.python.org/) (v3.9 or later)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/whiteboard-processor.git
cd whiteboard-processor
```

2. Install dependencies:

```bash
npm install -g serverless
npm install
pip install -r requirements.txt
```

3. Deploy to AWS:

```bash
serverless deploy --stage dev
```

### Configuration

The system can be configured using environment variables:

- `STAGE`: The deployment stage (e.g., `dev`, `staging`, `prod`).
- `STORAGE_BUCKET`: The S3 bucket for storing videos, frames, and results.
- `STATE_MACHINE_ARN`: The ARN of the Step Functions state machine.

## Usage

### API Endpoints

#### POST /process

Upload a video for processing.

**Request:**

```
POST /process
Content-Type: multipart/form-data

file: <video_file>
context: {
  "codebase_context": "Optional context about the codebase",
  "project_name": "Optional project name",
  "additional_info": {
    "key": "value"
  }
}
```

**Response:**

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing"
}
```

#### GET /status/{job_id}

Get the status of a processing job.

**Response:**

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "steps": {
    "frame_extraction": {
      "status": "completed",
      "frames_count": 100,
      "frames_key": "metadata/123e4567-e89b-12d3-a456-426614174000/frames.json"
    },
    "audio_extraction": {
      "status": "completed",
      "audio_key": "audio/123e4567-e89b-12d3-a456-426614174000/audio.wav"
    },
    "content_detection": {
      "status": "completed",
      "regions_count": 50,
      "regions_key": "metadata/123e4567-e89b-12d3-a456-426614174000/regions.json"
    },
    "content_tracking": {
      "status": "completed",
      "tracks_count": 10,
      "tracks_key": "metadata/123e4567-e89b-12d3-a456-426614174000/tracks.json"
    },
    "transcription": {
      "status": "completed",
      "transcript_key": "metadata/123e4567-e89b-12d3-a456-426614174000/transcript.json"
    },
    "content_analysis": {
      "status": "completed",
      "analysis_key": "metadata/123e4567-e89b-12d3-a456-426614174000/analysis.json"
    },
    "output_generation": {
      "status": "completed",
      "result_key": "results/123e4567-e89b-12d3-a456-426614174000.json"
    }
  }
}
```

#### GET /result/{job_id}

Get the result of a completed processing job.

**Response:**

```json
{
  "items": [
    {
      "track_id": 1,
      "frames": [0, 30, 60, 90],
      "summary": "This is a summary of the content in track 1."
    },
    {
      "track_id": 2,
      "frames": [120, 150, 180, 210],
      "summary": "This is a summary of the content in track 2."
    }
  ]
}
```

## Development

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API locally:

```bash
cd hackillinois25
uvicorn orchestrator:app --reload
```

3. Test the API:

```bash
curl -X POST -F "file=@path/to/video.mp4" http://localhost:8000/process
```

### Adding New Components

To add a new component to the pipeline:

1. Create a new handler in the `handlers` directory.
2. Add the handler to the `serverless.yml` file.
3. Update the Step Functions state machine to include the new step.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
