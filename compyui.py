"""
Standalone ComfyUI Workflow Executor
Run ComfyUI workflows as pure Python scripts without server/API

Based on: https://www.timlrx.com/blog/executing-comfyui-workflows-as-standalone-scripts

This script executes ComfyUI workflows directly by:
1. Loading the workflow JSON (API format)
2. Executing nodes in the correct order
3. Managing outputs without any web server

Requirements:
- ComfyUI installed (for its core modules)
- All custom nodes installed that your workflow uses
- Models downloaded to correct directories
"""

import sys
import os
import json
from typing import Dict, List, Any, Tuple
from collections import OrderedDict

# Add ComfyUI to Python path
COMFYUI_PATH = "/path/to/ComfyUI"  # UPDATE THIS!
sys.path.insert(0, COMFYUI_PATH)

# Import ComfyUI core modules
import execution
import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_custom_sampler
import folder_paths


class ExecutionCache:
    """
    Manages caching of node outputs during workflow execution
    """
    def __init__(self):
        self.outputs = {}
        self.ui = {}
        self.objects_to_delete = []
    
    def get(self, node_id: str) -> Tuple[Any, Any]:
        """Get cached output for a node"""
        return self.outputs.get(node_id, None), self.ui.get(node_id, None)
    
    def set(self, node_id: str, output: Any, ui: Any = None):
        """Cache output for a node"""
        self.outputs[node_id] = output
        if ui is not None:
            self.ui[node_id] = ui
    
    def clear(self):
        """Clear all cached outputs"""
        self.outputs.clear()
        self.ui.clear()


class WorkflowExecutor:
    """
    Executes ComfyUI workflows as standalone Python scripts
    """
    def __init__(self, workflow_path: str, verbose: bool = True):
        """
        Initialize workflow executor
        
        Args:
            workflow_path: Path to workflow_api.json file
            verbose: Print execution progress
        """
        self.workflow_path = workflow_path
        self.verbose = verbose
        self.cache = ExecutionCache()
        
        # Load workflow
        with open(workflow_path, 'r') as f:
            self.workflow = json.load(f)
        
        if self.verbose:
            print(f"Loaded workflow from: {workflow_path}")
            print(f"Total nodes: {len(self.workflow)}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow and return all outputs
        
        Returns:
            Dictionary mapping node_id -> output
        """
        # Validate workflow
        valid, error, _, _ = execution.validate_prompt(self.workflow)
        if not valid:
            raise ValueError(f"Invalid workflow: {error}")
        
        if self.verbose:
            print("\nStarting workflow execution...")
        
        # Create execution list
        prompt = execution.DynamicPrompt(self.workflow)
        execution_list = execution.ExecutionList(prompt, folder_paths.get_output_directory())
        
        # Execute each node in order
        for node_id, class_type, is_output in execution_list.to_execute:
            self._execute_node(node_id, class_type, is_output)
        
        if self.verbose:
            print("\nWorkflow execution complete!")
        
        return self.cache.outputs
    
    def _execute_node(self, node_id: str, class_type: str, is_output: bool):
        """
        Execute a single node in the workflow
        
        Args:
            node_id: Unique identifier for the node
            class_type: Type of node (e.g., "KSampler", "VAEDecode")
            is_output: Whether this is an output node
        """
        if self.verbose:
            print(f"Executing node {node_id}: {class_type}")
        
        # Get node class
        node_class = NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            raise ValueError(f"Unknown node type: {class_type}")
        
        # Get input data for this node
        inputs = execution.get_input_data(
            self.workflow[node_id]["inputs"],
            class_type,
            node_id,
            self.cache.outputs
        )
        
        # Execute node
        obj = node_class()
        output_data, output_ui = execution.get_output_data(obj, inputs)
        
        # Cache results
        self.cache.set(node_id, output_data, output_ui)
        
        # Handle output nodes (save images, videos, etc.)
        if is_output:
            self._handle_output(node_id, class_type, output_data, output_ui)
    
    def _handle_output(self, node_id: str, class_type: str, output_data: Any, output_ui: Any):
        """
        Handle output from output nodes (SaveImage, etc.)
        
        Args:
            node_id: Node identifier
            class_type: Type of output node
            output_data: Output tensor/data
            output_ui: UI data (filenames, etc.)
        """
        if self.verbose:
            print(f"  └─ Output node completed")
        
        if output_ui and "images" in output_ui:
            for img_info in output_ui["images"]:
                filename = img_info.get("filename", "unknown")
                subfolder = img_info.get("subfolder", "")
                print(f"     Saved: {os.path.join(subfolder, filename)}")


def run_vid2vid_workflow(
    workflow_path: str,
    video_path: str = None,
    style_image_path: str = None,
    prompt: str = None,
    negative_prompt: str = None,
    skip_frames: int = 0,
    max_frames: int = 100,
    output_dir: str = None
):
    """
    Run Vid2Vid Part 2 workflow with custom parameters
    
    Args:
        workflow_path: Path to workflow_api.json
        video_path: Path to input video
        style_image_path: Path to style reference image
        prompt: Positive prompt
        negative_prompt: Negative prompt
        skip_frames: Number of frames to skip at start
        max_frames: Maximum number of frames to process
        output_dir: Custom output directory
    """
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Update parameters in workflow
    # NOTE: You'll need to identify the correct node IDs from your workflow
    # This is an example structure - adjust based on your actual workflow
    
    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        # Update video input
        if class_type == "LoadVideo" and video_path:
            inputs["video"] = video_path
            print(f"Updated video input: {video_path}")
        
        # Update style image
        if class_type == "LoadImage" and style_image_path:
            if "image" in inputs:
                inputs["image"] = style_image_path
                print(f"Updated style image: {style_image_path}")
        
        # Update prompts
        if class_type == "CLIPTextEncode":
            if prompt and "positive" in str(node_data).lower():
                inputs["text"] = prompt
                print(f"Updated positive prompt")
            elif negative_prompt and "negative" in str(node_data).lower():
                inputs["text"] = negative_prompt
                print(f"Updated negative prompt")
        
        # Update frame parameters
        if "skip_first_frames" in inputs and skip_frames is not None:
            inputs["skip_first_frames"] = skip_frames
            print(f"Set skip_frames: {skip_frames}")
        
        if "frame_load_cap" in inputs and max_frames is not None:
            inputs["frame_load_cap"] = max_frames
            print(f"Set max_frames: {max_frames}")
    
    # Save modified workflow temporarily
    temp_workflow_path = "temp_workflow_api.json"
    with open(temp_workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    # Set output directory
    if output_dir:
        folder_paths.set_output_directory(output_dir)
    
    # Execute workflow
    executor = WorkflowExecutor(temp_workflow_path, verbose=True)
    outputs = executor.execute()
    
    # Clean up temp file
    os.remove(temp_workflow_path)
    
    return outputs


def batch_process_long_video(
    workflow_path: str,
    video_path: str,
    total_frames: int,
    batch_size: int = 100,
    **kwargs
):
    """
    Process a long video in batches to avoid memory issues
    
    Args:
        workflow_path: Path to workflow
        video_path: Path to input video
        total_frames: Total number of frames in video
        batch_size: Frames per batch (max 100 recommended)
        **kwargs: Additional parameters for run_vid2vid_workflow
    """
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    print(f"Processing {total_frames} frames in {num_batches} batches")
    print(f"Batch size: {batch_size} frames\n")
    
    all_outputs = []
    
    for batch_num in range(num_batches):
        skip_frames = batch_num * batch_size
        remaining_frames = total_frames - skip_frames
        max_frames = min(batch_size, remaining_frames)
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num + 1}/{num_batches}")
        print(f"Frames: {skip_frames} to {skip_frames + max_frames - 1}")
        print(f"{'='*60}\n")
        
        outputs = run_vid2vid_workflow(
            workflow_path=workflow_path,
            video_path=video_path,
            skip_frames=skip_frames,
            max_frames=max_frames,
            output_dir=f"output_batch_{batch_num:03d}",
            **kwargs
        )
        
        all_outputs.append(outputs)
    
    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE!")
    print(f"{'='*60}")
    
    return all_outputs


def main():
    """
    Example usage
    """
    # UPDATE THIS PATH!
    workflow_path = "workflow_api.json"
    
    # For videos under 100 frames
    outputs = run_vid2vid_workflow(
        workflow_path=workflow_path,
        video_path="input_video.mp4",
        style_image_path="style_reference.png",
        prompt="cinematic, dramatic lighting, professional quality",
        negative_prompt="blurry, low quality, distorted",
        max_frames=100
    )
    
    # For longer videos (300 frames example)
    # batch_process_long_video(
    #     workflow_path=workflow_path,
    #     video_path="long_video.mp4",
    #     total_frames=300,
    #     batch_size=100,
    #     style_image_path="style_reference.png",
    #     prompt="cinematic, dramatic lighting",
    #     negative_prompt="blurry, low quality"
    # )


if __name__ == "__main__":
    """
    Setup instructions:
    
    1. Install ComfyUI:
       git clone https://github.com/comfyanonymous/ComfyUI.git
       cd ComfyUI
       pip install -r requirements.txt
    
    2. Install custom nodes needed for Vid2Vid Part 2:
       - AnimateDiff
       - ControlNet
       - IPAdapter
       - Video nodes
    
    3. Download models to appropriate directories:
       - models/checkpoints/
       - models/controlnet/
       - models/animatediff/
       - models/ipadapter/
    
    4. Update COMFYUI_PATH at the top of this script
    
    5. Export your workflow as API format:
       - Open ComfyUI web interface
       - Enable Dev Mode in settings
       - Save (API Format)
    
    6. Run this script:
       python standalone_executor.py
    """
    
    # Update the path at the top of this file!
    if COMFYUI_PATH == "/path/to/ComfyUI":
        print("ERROR: Please update COMFYUI_PATH at the top of this script!")
        print("Set it to your actual ComfyUI installation directory.")
        sys.exit(1)
    
    main()