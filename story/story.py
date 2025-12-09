"""
Curify Storyboard Labeling & Scene Analysis Pipeline
A comprehensive system for automatic video scene segmentation, 
shot labeling, and transition detection.
"""
import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple

# Import local modules
from models import Scene, Shot, Transition, ShotType, CameraMove, TransitionType, Storyboard
from video_analyzer import VideoAnalyzer
from storyboard_generator import StoryboardGenerator
from youtube_utils import is_youtube_url, download_youtube_video
from object_detector import Detection
from ai_analyzer import OllamaAnalyzer, create_analyzer
from advanced_camera_analyzer import AdvancedCameraAnalyzer as CameraAnalyzer

def process_video(video_path: str, output_path: str, threshold: float = 0.4, use_ai: bool = True):
    """
    Process a video and generate a storyboard
    
    Args:
        video_path: Path to the input video file or YouTube URL
        output_path: Path to save the output JSON file
        threshold: Scene detection threshold (0-1)
        use_ai: Whether to use AI for enhanced analysis
    """
    # Check if it's a YouTube URL
    if is_youtube_url(video_path):
        print(f"Downloading YouTube video: {video_path}")
        video_path = download_youtube_video(video_path)
        print(f"Downloaded to: {video_path}")
    
    # Initialize video analyzer
    print(f"Analyzing video: {video_path}")
    analyzer = VideoAnalyzer(video_path)
    
    # Initialize Camera Analyzer
    print("Initializing advanced camera analyzer...")
    camera_analyzer = CameraAnalyzer(
        frame_width=analyzer.width,
        frame_height=analyzer.height,
        fps=analyzer.fps
    )
    
    # Initialize AI analyzer if requested
    ai_analyzer = None
    if use_ai:
        try:
            print("Initializing Ollama analyzer...")
            ai_analyzer = create_analyzer(model="llava:latest")
            if not ai_analyzer.available:
                print("Warning: Ollama is not available. Falling back to basic analysis.")
                use_ai = False
            else:
                print("Ollama analyzer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize AI analyzer: {e}")
            use_ai = False
    
    storyboard = StoryboardGenerator(analyzer, ai_analyzer=ai_analyzer, camera_analyzer=camera_analyzer)
    
    # Process video and generate scenes
    print("Processing video and generating storyboard...")
    scenes = storyboard.analyze_video(threshold=threshold, use_ai=use_ai)
    
    # Save results
    print(f"Saving results to: {output_path}")
    
    # Create a Storyboard instance with the generated scenes
    storyboard = Storyboard(
        scenes=scenes,
        metadata={
            'video_path': video_path,
            'duration': analyzer.duration,
            'resolution': f"{analyzer.width}x{analyzer.height}",
            'fps': analyzer.fps,
            'frame_count': analyzer.frame_count
        }
    )
    
    # Save to JSON
    storyboard.to_json(output_path)
    print("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description="Video Storyboard Generator with AI Analysis")
    parser.add_argument("input", help="Input video file or YouTube URL")
    parser.add_argument("-o", "--output", default="storyboard.json", 
                       help="Output JSON file (default: storyboard.json)")
    parser.add_argument("-t", "--threshold", type=float, default=0.4,
                       help="Scene detection threshold (0-1, default: 0.4)")
    parser.add_argument("--table", action="store_true",
                       help="Export storyboard as a readable table")
    parser.add_argument("--no-ai", action="store_false", dest="use_ai",
                       help="Disable AI analysis (faster but less detailed)")
    
    args = parser.parse_args()
    
    try:
        process_video(args.input, args.output, args.threshold, args.use_ai)
        
        # If table output is requested, generate a table version
        if args.table:
            table_path = args.output.replace('.json', '.txt')
            with open(args.output, 'r') as f:
                data = json.load(f)
            
            with open(table_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("STORYBOARD SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                for scene in data.get('scenes', []):
                    f.write(f"SCENE {scene.get('scene_id', 'N/A')}\n")
                    f.write("-" * 80 + "\n")
                    metadata = scene.get('scene_metadata', {})
                    f.write(f"Time: {metadata.get('start_time', 0):.2f}s - {metadata.get('end_time', 0):.2f}s\n")
                    f.write(f"Duration: {metadata.get('duration_seconds', 0):.2f}s\n")
                    f.write(f"Mood: {metadata.get('dominant_mood', 'N/A')}\n")
                    f.write(f"Environment: {metadata.get('environment', 'N/A')}\n")
                    
                    # Write shots if available
                    shots = scene.get('shots', [])
                    if shots:
                        f.write("\nShots:\n")
                        for shot in shots:
                            f.write(f"  - {shot.get('shot_type', 'N/A')} | "
                                  f"{shot.get('camera_move', 'N/A')} | "
                                  f"{shot.get('visual_focus', 'N/A')}\n")
                    
                    # Write description if available
                    description = metadata.get('description')
                    if description:
                        f.write("\nDescription: " + description + "\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
            
            print(f"\nTable version saved to: {table_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()