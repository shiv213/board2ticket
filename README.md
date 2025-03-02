# board2ticket - HackIllinois25 **Winners!!!**

> Transforming whiteboard discussions into structured GitHub tickets automatically.

## 📋 Overview

board2ticket is an innovative software solution developed by Shiv Trivedi, Ahaan Kanaujia, and Aditya Kunte at Hack Illinois 2025. It automatically transforms whiteboard discussions about code issues into structured GitHub tickets by leveraging advanced computer vision, audio processing, and natural language processing technologies.

## 🔍 Problem Statement

Software development teams often use whiteboards for collaborative problem-solving and design discussions. However, valuable insights and decisions from these sessions are frequently lost in the transition to actionable development tasks, leading to:

- Information loss between ideation and implementation
- Time wasted on manual note-taking and ticket creation
- Difficulty maintaining context between discussions and code
- Limited historical record of design decisions

## 💡 Solution

board2ticket bridges the gap between collaborative whiteboard sessions and actionable development tasks through a comprehensive multimodal pipeline:

1. **Capture whiteboard sessions** (video + audio)
2. **Process visual content** using advanced computer vision techniques
3. **Transcribe and analyze discussions** using audio processing
4. **Align visual and audio data** through temporal mapping
5. **Generate structured GitHub tickets** with all relevant context

## 🛠️ Technical Implementation

### Image Processing Pipeline

- **Frame Extraction and Preprocessing**
  - Convert video frames to grayscale
  - Apply binary thresholding (threshold value 130)
  - Generate inverted binary images to highlight content

- **Content Detection**
  - OpenCV contour detection (cv2.findContours with RETR_EXTERNAL mode)
  - Filter contours based on minimum area (50px²)
  - Extract precise bounding boxes

- **Intelligent Region Clustering**
  - DBSCAN clustering to group related content
  - Centroid calculation for content elements
  - Proximate element merging (eps=100)
  - Oversized region filtering (>40% frame width or >50% frame height)

- **Cluster Tracking**
  - Sequential frame processing
  - Pixel density change monitoring
  - Content update timestamping

### Audio Processing Pipeline

- **Silence-Based Segmentation**
  - pydub's split_on_silence (-40dBFS threshold)
  - 1500ms minimum silence for topic transitions
  - Precise timestamp generation

- **Speech-to-Text Conversion**
  - OpenAI's Whisper model for transcription
  - Timestamp-linked transcriptions

- **Semantic Clustering**
  - Text embeddings via OpenAI's text-embedding-3-small
  - Unsupervised clustering (Agglomerative or K-means)
  - Topic cluster identification
  - Temporal sequence preservation

### Multimodal Integration

- **Temporal Alignment** of whiteboard updates with audio segments
- **Context Enrichment** with codebase information
- **Vision-Language Modeling** for combined data processing

### Ticket Generation

- **Content Clustering** for related discussions
- **Metadata Generation** for structured ticket fields
- **Visual Reference Inclusion** for context
- **Codebase Integration** for implementation guidance

## 🔄 System Pipeline

1. **Codebase Contextualization**: Converting GitHub repositories into LLM-friendly format
2. **Video Processing**: Custom OpenCV-based pipeline for text and diagrams
3. **Audio Processing**: Segmenting and transcribing with pydub and OpenAI API
4. **Multimodal Summarization**: Vision-language model interpretation
5. **Temporal Clustering**: Timeline-based discussion grouping
6. **Ticket Generation**: Structured GitHub issues creation

## ✨ Benefits

- **Meeting Efficiency**: Eliminates manual note-taking and ticket creation
- **Reduced Information Loss**: Captures all whiteboard content and discussions
- **Improved Development Workflow**: Creates clear, contextual tickets
- **Enhanced Collaboration**: Preserves collaborative intelligence
- **Historical Record**: Maintains visual and textual development history

## 📚 Technical Requirements

- Python 3.x
- OpenCV (cv2) for image processing
- NumPy and scikit-learn for data processing and clustering
- PyDub and librosa for audio processing
- OpenAI API (Whisper transcription and text embeddings)
- Vision-language models for multimodal processing
- LLM integration for ticket generation

## 👥 Team

- Shiv Trivedi
- Ahaan Kanaujia
- Aditya Kunte

## 🏆 Recognition

Developed at Hack Illinois 2025 at the University of Illinois Urbana Champaign.
