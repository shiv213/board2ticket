import io
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
import soundfile as sf
import librosa
import librosa.effects
import ffmpeg
from pydub import AudioSegment
from pydub.silence import split_on_silence

from scipy.io.wavfile import write
from IPython.display import Audio

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

open_ai_key = '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'

from openai import OpenAI
client = OpenAI(api_key=open_ai_key)

def extract_audio_from_mkv(input_mkv, output_wav, output_dir):
    """
    Extracts audio from an MKV video file and saves it as a WAV file.

    Parameters:
    - input_mkv (str): Path to the input MKV video file
    - output_wav (str): Path where the extracted WAV audio will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        (
            ffmpeg
            .input(input_mkv)
            .output(output_wav, format='wav', acodec='pcm_s16le', ac=1, ar=16000)  # Convert to 16kHz mono PCM WAV
            .run(overwrite_output=True)
        )
        print(f"Audio extracted successfully: {output_wav}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def split_audio_by_pauses(audio_path, silence_thresh=-40, min_silence_len=1500):
    """
    Splits the audio file into chunks based on pauses and returns timestamps.

    Parameters:
    - audio_path (str): Path to the input audio file
    - silence_thresh (int): Silence threshold in dBFS. Default is -40 dBFS
    - min_silence_len (int): Minimum silence length (milliseconds) to consider as a pause. Default is 1500 ms

    Returns:
    - List of tuples: (AudioSegment, start_time, end_time)
    """
    audio = AudioSegment.from_wav(audio_path)

    # Split audio based on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True
    )

    timestamped_chunks = []
    current_time = 0  # Track start time

    for chunk in chunks:
        start_time = current_time
        end_time = start_time + len(chunk)  # End time = start time + chunk duration
        timestamped_chunks.append((chunk, start_time, end_time))
        current_time = end_time  # Move to next segment

    return timestamped_chunks

def save_chunks_with_timestamps(chunks, output_dir="./data", base_filename="chunk"):
    """
    Saves chunks as .wav files and logs their timestamps.

    Parameters:
    - chunks (list of tuples): List containing (AudioSegment, start_time, end_time)
    - output_dir (str): Directory to save chunk files
    - base_filename (str): Base name for chunk files

    Returns:
    - List of dictionaries with filenames and timestamps
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_info = []

    for i, (chunk, start_time, end_time) in enumerate(chunks):
        chunk_filename = f"{output_dir}/{base_filename}_{i+1}.wav"
        chunk.export(chunk_filename, format="wav")

        chunk_info.append({
            "filename": chunk_filename,
            "start_time": start_time,
            "end_time": end_time
        })

    return chunk_info

def transcribe_audio(file_path):
    """
    Transcribes the given audio file to text using OpenAI's Whisper model.

    Parameters:
    - file_path (str): Path to the audio file to be transcribed

    Returns:
    - str: Transcribed text from the audio file
    """
    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",  # Using the whisper-1 model
            file=open(file_path, "rb")
        )
        return response.text
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

def get_transcript_of_chunks(chunk_files):
    """
    Retrieves transcriptions for a list of audio chunk files.

    Parameters:
    - chunk_files (list): List of file paths to chunked audio files

    Returns:
    - Dictionary mapping chunk indices to transcriptions
    """
    audio_segment_transcripts = {}
    for i, filename in enumerate(chunk_files):
        file_path = os.path.join("./data", filename)
        transcription = transcribe_audio(file_path)
        transcript_filename = os.path.join("./data", f"transcript_{i+1}.txt")
        try:
          with open(transcript_filename, 'w') as transcript_file:
              transcript_file.write(transcription)
          audio_segment_transcripts[i + 1] = transcription
        except Exception as e:
          print(e)
    return audio_segment_transcripts

def get_embeddings(text):
    """
    Generates embeddings for the given text using OpenAI's embeddings model.

    Parameters:
    - text (str): The text to generate embeddings for

    Returns:
    - list: Embedding vector representing the input text
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Embedding model
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

def cluster_audio_segments(audio_segment_transcripts, num_clusters, method="agglomerative"):
    """
    Clusters transcriptions into a fixed number of clusters using KMeans or Agglomerative Clustering.

    Parameters:
        - audio_segment_transcripts: Dictionary where keys are segment IDs (int) and values are transcriptions (str)
        - num_clusters: Fixed number of clusters
        - method: Clustering method ("kmeans" or "agglomerative")

    Returns:
        - Dictionary with cluster IDs as keys and lists of segment IDs as values
    """
    if not audio_segment_transcripts:
        return {}

    segment_ids = list(audio_segment_transcripts.keys())
    transcriptions = list(audio_segment_transcripts.values())

    # Get embeddings for each transcription
    embeddings = []
    for transcription in transcriptions:
        embedding = get_embeddings(transcription)
        if embedding is not None:
            embeddings.append(embedding)

    if len(embeddings) == 0:
        print("No embeddings generated. Cannot perform clustering.")
        return {}

    embeddings_array = np.array(embeddings)

    # Perform Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=num_clusters)
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans' or 'agglomerative'.")

    labels = model.fit_predict(embeddings_array)

    # Assign segment IDs to clusters
    clustered_segments = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(labels):
        clustered_segments[label].append(segment_ids[idx])
    return clustered_segments

def find_relevant_audio_clips(chunk_info, time_ranges):
    """
    Finds audio clips that match the specified time ranges.

    Parameters:
        - chunk_info (list of dict): List of dictionaries with 'file', 'start', 'end' times (in ms)
        - time_ranges (list of tuples): List of (start_time, end_time) tuples (in seconds)

    Returns:
        - Dictionary mapping each (start_time, end_time) to a list of matching files
    """
    relevant_clips = {}

    for start_time, end_time in time_ranges:
        start_time_ms = start_time * 1000
        end_time_ms = end_time * 1000
        matching_files = []
        for info in chunk_info:
            filename = info['filename']
            chunk_start_time = info['start_time']
            chunk_end_time = info['end_time']
            if start_time_ms <= chunk_start_time <= end_time_ms or start_time_ms <= chunk_end_time <= end_time_ms:
                matching_files.append(filename)
        relevant_clips[(start_time, end_time)] = matching_files

    return relevant_clips

def combine_audio_clips(matching_files, output_directory="./output"):
    """
    Combines audio clips within each time range and saves the output as a single audio file per time range.

    Parameters:
        - matching_files (dict): Dictionary where keys are (start_time, end_time) tuples and values are lists of file paths.
        - output_directory (str): Directory where the combined audio files will be saved.

    Returns:
        - None (Saves the combined files to the output directory)
    """
    for (start_time, end_time), files in matching_files.items():
        combined = AudioSegment.empty()
        for file in files:
            audio = AudioSegment.from_wav(file)
            combined += audio

        output_path = os.path.join(output_directory, f"combined_{start_time}_{end_time}.wav")
        combined.export(output_path, format="wav")
        print(f"Saved combined audio for ({start_time}, {end_time}) to {output_path}")

def main(input_mkv, output_audio_dir="./data", num_clusters=5, silence_thresh=-40, min_silence_len=1500):
    """
    Main function to process an MKV video file, extract audio, split by pauses, transcribe the chunks,
    generate embeddings, and cluster the segments.

    Parameters:
    - input_mkv (str): Path to the MKV video file
    - output_audio_dir (str): Directory to save audio chunks and transcriptions
    - num_clusters (int): Number of clusters for clustering the audio segments
    - silence_thresh (int): Silence threshold in dBFS. Default is -40 dBFS
    - min_silence_len (int): Minimum silence length (milliseconds) to consider as a pause. Default is 1500 ms
    """
    print("Starting the process...")

    # Step 1: Extract audio from the MKV file
    output_wav = os.path.join(output_audio_dir, "extracted_audio.wav")
    extract_audio_from_mkv(input_mkv, output_wav, output_audio_dir)

    # Step 2: Split the audio into chunks based on pauses
    print("Splitting the audio into chunks...")
    timestamped_chunks = split_audio_by_pauses(output_wav, silence_thresh, min_silence_len)

    # Step 3: Save the chunks to files and log their timestamps
    print("Saving audio chunks...")
    # Define base filename from input MKV file name (without extension)
    base_filename = os.path.splitext(os.path.basename(input_mkv))[0]
    chunk_info = save_chunks_with_timestamps(timestamped_chunks, output_dir=output_audio_dir, base_filename=base_filename)

    # Step 4: Transcribe the chunks to text
    print("Transcribing audio chunks...")
    chunk_files = [f"{base_filename}_{i+1}.wav" for i in range(len(chunk_info))]
    audio_segment_transcripts = get_transcript_of_chunks(chunk_files)

    # Step 5: Cluster the audio segments based on their transcriptions
    print(f"Clustering the audio segments into {num_clusters} clusters...")
    clustered_segments = cluster_audio_segments(audio_segment_transcripts, num_clusters)

    # print some information on the clusters
    for cluster_id, segments in clustered_segments.items():
      print(f"\nCluster {cluster_id+1}:")
      for key in segments:
          print(audio_segment_transcripts[key])

    print("Process completed!")

if __name__ == "__main__":
  input_movie_file = "ahaan.mkv"
  main(input_movie_file)

