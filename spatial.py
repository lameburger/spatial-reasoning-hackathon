########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample shows how to capture a real-time 3D reconstruction      
    of the scene using the Spatial Mapping API. The resulting mesh      
    is displayed as a wireframe on top of the left image using OpenGL.  
    Spatial Mapping can be started and stopped with the Space Bar key
    
    RGB COLORS: By default, RGB colors are saved as per-vertex colors in the OBJ file.
    Use --save_texture flag to save as separate texture image files instead.
    
    YOLO HAZARD DETECTION: Integrates YOLO object detection to place hazards as
    labeled 3D points in space using depth information from the ZED camera.
"""
import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import tempfile
import os


# YOLO hazard class mappings
CONSTRUCTION_HAZARD_CLASSES = {
    'person': 'person',
    'machinery': 'excavator',
    'vehicle': 'dump_truck',
    'hardhat': 'hardhat',
    'mask': 'mask',
    'no-hardhat': 'no_hardhat',
    'no-mask': 'no_mask',
    'no-safety vest': 'no_safety_vest',
    'safety cone': 'safety_cone',
    'safety vest': 'safety_vest',
}

COCO_TO_HAZARD = {
    'person': 'person',
    'truck': 'dump_truck',
    'car': 'dump_truck',
    'bus': 'dump_truck',
}


def map_class_name(class_name: str) -> str:
    """Map detected class to hazard class. Only returns PPE violations."""
    class_name_lower = class_name.lower()
    
    # ONLY track PPE violations - no hardhat and no safety vest
    if 'no-hardhat' in class_name_lower or 'no hardhat' in class_name_lower or 'no_hardhat' in class_name_lower:
        return 'no_hardhat'
    if 'no-safety vest' in class_name_lower or 'no safety vest' in class_name_lower or 'no_safety_vest' in class_name_lower:
        return 'no_safety_vest'
    
    # Ignore everything else
    return None


def get_3d_position(point_cloud, x, y, width, height, sample_radius=5):
    """Get 3D position from point cloud at image coordinates."""
    err, point_cloud_value = point_cloud.get_value(x, y)
    
    if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point_cloud_value[0]) and not np.isinf(point_cloud_value[0]):
        return point_cloud_value[:3]
    
    # Sample nearby points if center is invalid
    for radius in range(1, sample_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                new_x = min(max(0, x + dx), width - 1)
                new_y = min(max(0, y + dy), height - 1)
                err, point_cloud_value = point_cloud.get_value(new_x, new_y)
                if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point_cloud_value[0]) and not np.isinf(point_cloud_value[0]):
                    return point_cloud_value[:3]
    
    return None


class HazardPoint:
    """Represents a detected hazard in 3D space."""
    def __init__(self, position, class_name, confidence, timestamp):
        self.position = position
        self.class_name = class_name
        self.confidence = confidence
        self.timestamp = timestamp
        self.lifetime = 2.0
    
    def is_expired(self, current_time):
        return (current_time - self.timestamp) > self.lifetime


class VLMAnalyzer:
    """Vision Language Model analyzer for construction safety"""
    def __init__(self, adapter_path="./vlm_fine_tuned/smolvlm_construction_finetuned"):
        self.model = None
        self.processor = None
        self.device = None  # Will be set when model loads
        self.adapter_path = adapter_path
        
    def load_model(self):
        """Load VLM model and processor with GPU optimizations"""
        try:
            import torch
            from transformers import Idefics3Processor, Idefics3ForConditionalGeneration
            from peft import PeftModel
            from PIL import Image
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                print("[VLM] WARNING: CUDA not available, running on CPU (slow)")
                self.device = "cpu"
            else:
                self.device = "cuda"
                # Enable GPU optimizations
                torch.cuda.set_per_process_memory_fraction(0.95)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.enabled = True
                torch.cuda.empty_cache()
            
            model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            
            print("[VLM] Loading model...")
            self.processor = Idefics3Processor.from_pretrained(model_id)
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()
            
            print("[VLM] Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[VLM] Failed to load model: {e}")
            return False
    
    def analyze_frame(self, frame_bgr, detected_hazards=None):
        """Analyze a frame for safety violations
        
        Args:
            frame_bgr: The frame to analyze (BGR format)
            detected_hazards: List of hazard class names detected by YOLO (optional)
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            import torch
            from PIL import Image
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Build context-aware prompt
            context = ""
            if detected_hazards:
                hazard_list = ", ".join(set(detected_hazards))
                context = f"YOLO detected: {hazard_list}\n"
            
            # Simplified, direct prompt
            prompt = f"""Analyze this construction site image for safety violations.

{context}
Look for:
- Missing hard hats
- Missing safety vests
- Workers near heavy equipment
- Fall hazards

Describe what you see and any safety concerns."""
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=[image], text=text, return_tensors="pt")
            
            # Move inputs to GPU/model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with optimizations
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            num_beams=1,
                            use_cache=True,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"[VLM] Analysis error: {e}")
            return None


def main(opt):
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # OpenGL's coordinate system is right_handed    
    init.depth_maximum_distance = 8.
    parse_args(init, opt)
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    camera_infos = zed.get_camera_information()
    pose = sl.Pose()
    
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        exit()
    
    # Configure spatial mapping with RGB capture
    if opt.build_mesh:
        spatial_mapping_parameters = sl.SpatialMappingParameters(
            resolution = sl.MAPPING_RESOLUTION.MEDIUM,
            mapping_range = sl.MAPPING_RANGE.MEDIUM,
            max_memory_usage = 2048,
            save_texture = opt.save_texture,  # RGB: False = per-vertex colors, True = texture files
            use_chunk_only = True,
            reverse_vertex_order = False,
            map_type = sl.SPATIAL_MAP_TYPE.MESH
        )
        pymesh = sl.Mesh() 
    else:
        spatial_mapping_parameters = sl.SpatialMappingParameters(
            resolution = sl.MAPPING_RESOLUTION.MEDIUM,
            mapping_range = sl.MAPPING_RANGE.MEDIUM,
            max_memory_usage = 2048,
            save_texture = False,  # Point clouds always use per-vertex colors
            use_chunk_only = True,
            reverse_vertex_order = False,
            map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        )
        pymesh = sl.FusedPointCloud()

    print(f"Type: {'MESH' if opt.build_mesh else 'POINT CLOUD'}")
    if opt.build_mesh:
        print(f"RGB Mode: {'Texture files' if opt.save_texture else 'Per-vertex colors (default)'}")
    else:
        print(f"RGB Mode: Per-vertex colors")
    
    # Load VLM model if enabled
    vlm_analyzer = None
    if opt.enable_vlm:
        # VLM requires hazard detection to be enabled
        if not opt.enable_hazard_detection:
            opt.enable_hazard_detection = True
        
        vlm_analyzer = VLMAnalyzer(opt.vlm_adapter)
        if not vlm_analyzer.load_model():
            vlm_analyzer = None
    
    # Load YOLO model if enabled
    yolo_model = None
    model_names = None
    if opt.enable_hazard_detection:
        try:
            from ultralytics import YOLO
            
            if opt.yolo_model is None:
                possible_paths = [
                    Path("yolo/models/models/pt/best_yolo11n.pt"),
                    Path("models/models/pt/best_yolo11n.pt"),
                    Path("models/models/pt/best_yolo11s.pt"),
                    Path("yolov8n.pt"),
                ]
                for path in possible_paths:
                    if path.exists():
                        opt.yolo_model = str(path)
                        break
                if opt.yolo_model is None:
                    opt.yolo_model = "yolov8n.pt"
            

            
            yolo_model = YOLO(opt.yolo_model)
            model_names = yolo_model.names
        except Exception as e:
            print(f"YOLO initialization failed: {e}")
            opt.enable_hazard_detection = False
    
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    mapping_activated = False

    image = sl.Mat()  
    point_cloud = sl.Mat()
    pose = sl.Pose()
    
    # Define hazards file path
    hazards_file = "hazards_detected.json"
    
    # Hazard points tracking
    hazard_points = []
    all_hazards_detected = []  # Store all hazards for export
    vlm_responses = []  # Store VLM analysis responses
    
    # VLM analysis tracking
    last_vlm_analysis_time = 0
    vlm_analysis_interval = 5.0  # Run VLM analysis every 5 seconds max
    
    # Position tracking for camera pose
    camera_pose = sl.Pose()
    
    # Stats tracking
    hazard_counts = {}  # Running count of hazards by type
    last_stats_display = time.time()
    stats_display_interval = 5.0  # Show stats every 5 seconds

    viewer = gl.GLViewer()

    viewer.init(zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, pymesh, int(opt.build_mesh))
    
    last_call = time.time()
    frame_count = 0
    
    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) <= sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve point cloud for 3D positioning
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # Get camera pose relative to world frame
            tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            
            # Update pose data (used for projection of the mesh over the current image)
            zed.get_position(pose)
            
            # Run YOLO hazard detection
            if opt.enable_hazard_detection and yolo_model and frame_count % opt.skip_frames == 0:
                image_ocv = image.get_data()
                # Convert RGBA to RGB (YOLO expects 3 channels)
                if image_ocv.shape[2] == 4:
                    image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)
                results = yolo_model(image_ocv, conf=opt.conf_threshold, verbose=False)
                result = results[0]
                
                # Remove expired hazard points
                current_time = time.time()
                hazard_points = [p for p in hazard_points if not p.is_expired(current_time)]
                
                # Check if we should run VLM analysis
                run_vlm = False
                if opt.enable_vlm and vlm_analyzer and (current_time - last_vlm_analysis_time) > vlm_analysis_interval:
                    if result.boxes is not None and len(result.boxes) > 0:
                        run_vlm = True
                        last_vlm_analysis_time = current_time
                
                # Track detected hazards for VLM context
                detected_hazards_this_frame = []
                
                # Process detections
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        detected_class = model_names[class_id]
                        
                        hazard_class = map_class_name(detected_class)
                        
                        # Track detected hazards for VLM
                        if hazard_class:
                            detected_hazards_this_frame.append(hazard_class)
                        
                        if hazard_class:
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            
                            position_3d = get_3d_position(point_cloud, center_x, center_y,
                                                         image.get_width(), image.get_height())
                            
                            if position_3d is not None:
                                # Transform hazard position to world coordinates
                                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                                    # Get camera position in world frame
                                    py_translation = sl.Translation()
                                    cam_tx = camera_pose.get_translation(py_translation).get()[0]
                                    cam_ty = camera_pose.get_translation(py_translation).get()[1]
                                    cam_tz = camera_pose.get_translation(py_translation).get()[2]
                                    
                                    # Transform hazard position relative to camera into world frame
                                    world_x = cam_tx + position_3d[0]
                                    world_y = cam_ty + position_3d[1]
                                    world_z = cam_tz + position_3d[2]
                                    
                                    hazard = HazardPoint(
                                        position=position_3d,
                                        class_name=hazard_class,
                                        confidence=conf,
                                        timestamp=current_time
                                    )
                                    hazard_points.append(hazard)
                                    
                                    # Update hazard counts
                                    hazard_counts[hazard_class] = hazard_counts.get(hazard_class, 0) + 1
                                    
                                    # Store for export
                                    all_hazards_detected.append({
                                        'class': hazard_class,
                                        'camera_position': [float(position_3d[0]), float(position_3d[1]), float(position_3d[2])],
                                        'world_position': [float(world_x), float(world_y), float(world_z)],
                                        'confidence': float(conf),
                                        'timestamp': float(current_time),
                                        'frame': frame_count
                                    })
                                else:
                                    hazard = HazardPoint(
                                        position=position_3d,
                                        class_name=hazard_class,
                                        confidence=conf,
                                        timestamp=current_time
                                    )
                                    hazard_points.append(hazard)
                                    
                                    # Update hazard counts
                                    hazard_counts[hazard_class] = hazard_counts.get(hazard_class, 0) + 1
                                    
                                    # Store for export (camera frame only)
                                    all_hazards_detected.append({
                                        'class': hazard_class,
                                        'camera_position': [float(position_3d[0]), float(position_3d[1]), float(position_3d[2])],
                                        'world_position': None,
                                        'confidence': float(conf),
                                        'timestamp': float(current_time),
                                        'frame': frame_count
                                    })
                
                # Run VLM analysis after processing all detections
                if run_vlm and detected_hazards_this_frame:
                    print(f"\n{'='*70}")
                    print(f"VLM SAFETY ANALYSIS")
                    print(f"{'='*70}")
                    vlm_response = vlm_analyzer.analyze_frame(image_ocv, detected_hazards_this_frame)
                    
                    if vlm_response and len(vlm_response.strip()) > 0:
                        print(vlm_response)
                        # Store VLM response
                        vlm_responses.append({
                            'timestamp': float(current_time),
                            'frame': frame_count,
                            'hazards_detected': detected_hazards_this_frame,
                            'response': vlm_response
                        })
                    print(f"{'='*70}\n")
                
                # Display periodic stats
                if opt.enable_hazard_detection and (current_time - last_stats_display) > stats_display_interval:
                    if hazard_counts:
                        print(f"\n[STATS] Total hazards detected: {sum(hazard_counts.values())}")
                        for hazard_type, count in sorted(hazard_counts.items()):
                            print(f"  - {hazard_type}: {count}")
                        last_stats_display = current_time

            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                # Compute elapsed time since the last call of Camera.request_spatial_map_async()
                duration = time.time() - last_call
                # Ask for a mesh update if 500ms elapsed since last request
                if(duration > .5 and viewer.chunks_updated()):
                    zed.request_spatial_map_async()
                    last_call = time.time()

                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()

            change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)
            
            frame_count += 1

            if change_state:
                if not mapping_activated:
                    init_pose = sl.Transform()
                    zed.reset_positional_tracking(init_pose)

                    # Configure spatial mapping parameters
                    spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
                    spatial_mapping_parameters.use_chunk_only = True
                    spatial_mapping_parameters.save_texture = opt.save_texture if opt.build_mesh else False
                    if opt.build_mesh:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.MESH
                    else:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

                    # Enable spatial mapping
                    zed.enable_spatial_mapping(spatial_mapping_parameters)
                    print("\n=== Spatial Mapping STARTED ===")

                    # Clear previous mesh data
                    pymesh.clear()
                    viewer.clear_current_mesh()

                    # Start timer
                    last_call = time.time()

                    mapping_activated = True
                else:
                    print("\n=== Spatial Mapping STOPPED - Extracting mesh ===")
                    # Extract whole mesh
                    zed.extract_whole_spatial_map(pymesh)

                    if opt.build_mesh:
                        filter_params = sl.MeshFilterParameters()
                        filter_params.set(sl.MESH_FILTER.MEDIUM) 
                        # Filter the extracted mesh
                        pymesh.filter(filter_params, True)
                        viewer.clear_current_mesh()

                        # If textures have been saved during spatial mapping, apply them to the mesh
                        if opt.save_texture:
                            print("Applying texture files to mesh...")
                            pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)
                        else:
                            print("RGB colors saved as per-vertex colors in OBJ")

                    # Print statistics
                    if opt.build_mesh:
                        print(f"Mesh: {len(pymesh.vertices)} vertices, {len(pymesh.triangles)} triangles")
                    else:
                        print(f"Point cloud: {len(pymesh.vertices)} points")
                        print("RGB colors saved as per-vertex colors")

                    # Save mesh as an obj file
                    filepath = "mesh_gen.obj"
                    status = pymesh.save(filepath)
                    if status:
                        print(f"Saved to: {filepath}")
                        if opt.save_texture:
                            print("Texture image files saved alongside OBJ")
                    else:
                        print("Failed to save the mesh under " + filepath)
                    
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                    mapping_activated = False
    
    # Save hazard detections to JSON
    if opt.enable_hazard_detection and all_hazards_detected:
        with open(hazards_file, 'w') as f:
            json.dump({
                'total_hazards': len(all_hazards_detected),
                'total_frames': frame_count,
                'hazards': all_hazards_detected,
                'vlm_analyses': vlm_responses if vlm_responses else []
            }, f, indent=2)
        print(f"\nHazards saved to: {hazards_file}")
        if vlm_responses:
            print(f"VLM analyses saved: {len(vlm_responses)} responses")
    
    # Cleanup
    image.free(memory_type=sl.MEM.CPU)
    point_cloud.free()
    pymesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
   
          
def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--build_mesh', help = 'Either the script should plot a mesh or point clouds of surroundings', action='store_true')
    parser.add_argument('--save_texture', help='Save RGB as texture files (default: per-vertex colors)', action='store_true')
    
    # YOLO hazard detection arguments (ENABLED BY DEFAULT)
    parser.add_argument('--enable_hazard_detection', help='Enable YOLO hazard detection (default: enabled)', action='store_true', default=True)
    parser.add_argument('--disable_hazard_detection', help='Disable YOLO hazard detection', action='store_true')
    parser.add_argument('--yolo_model', type=str, default=None, help='Path to YOLO model (auto-detects if None)')
    parser.add_argument('--conf_threshold', type=float, default=0.6, help='YOLO confidence threshold (0.8 = high confidence, reduces false positives)')
    parser.add_argument('--skip_frames', type=int, default=2, help='Process YOLO every N frames (2 = every other frame for performance)')
    
    # VLM analysis arguments (ENABLED BY DEFAULT)
    parser.add_argument('--enable_vlm', help='Enable Vision-Language Model analysis for detailed safety assessment (default: enabled)', action='store_true', default=True)
    parser.add_argument('--disable_vlm', help='Disable VLM analysis', action='store_true')
    parser.add_argument('--vlm_adapter', type=str, default='./vlm_fine_tuned/smolvlm_construction_finetuned', 
                        help='Path to VLM fine-tuned adapter')
    
    opt = parser.parse_args()
    
    # Handle disable flags
    if opt.disable_hazard_detection:
        opt.enable_hazard_detection = False
    if opt.disable_vlm:
        opt.enable_vlm = False
    
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)

