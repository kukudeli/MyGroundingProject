"""
Visualize point cloud with ground truth and predicted bounding boxes using Open3D.

This script provides functions to:
1. Load point cloud from .bin or .npy files
2. Extract points inside bounding boxes
3. Visualize point cloud with GT and predicted bboxes
4. Save visualizations to PLY files

Color scheme:
    - Point cloud: Original grayscale intensity
    - Points inside GT bbox: Yellow
    - GT bbox edges: Red cylinders
    - Pred bbox edges: Green cylinders
"""

import numpy as np
import open3d as o3d
import json
import os
import sys
import random
import glob
# Add utils to path to import existing functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from box_util import rotz, extract_points_in_bbox_3d


def load_lidar_bin(pcd_path):
    """Load point cloud from binary file.
    Format: (x, y, z, intensity) - 4 dimensions
    Same as: pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4) if pcd_path.endswith(".bin") else np.load(pcd_path)
    """
    pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4) if pcd_path.endswith(".bin") else np.load(pcd_path)
    
    xyz = pcd[:, :3]
    intensity = pcd[:, 3]
    
    # Convert intensity to RGB for visualization (grayscale)
    # Normalize intensity to [0, 1] range
    intensity_normalized = np.tanh(intensity)  # Same as in dataset
    intensity_normalized = (intensity_normalized - intensity_normalized.min()) / (intensity_normalized.max() - intensity_normalized.min() + 1e-6)
    
    # Create RGB from intensity (grayscale)
    rgb = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=1)
    
    return pcd, xyz, rgb


def get_bbox_corners(bbox, has_yaw=True):
    """
    Get 8 corners of a 3D bounding box.
    bbox: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h]
    Returns: (8, 3) array of corner coordinates
    
    Corner order (same as box_util.py conver_box2d):
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
      
    x_corners: [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners: [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners: [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    """
    if has_yaw:
        x, y, z, l, w, h, yaw = bbox
    else:
        x, y, z, l, w, h = bbox
        yaw = 0.0
    
    # Create corners in local coordinate system (same order as box_util.py)
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
    
    corners = np.stack([x_corners, y_corners, z_corners], axis=1)  # (8, 3)
    
    # Rotate using rotz from box_util.py
    R = rotz(yaw)
    corners = corners @ R.T
    
    # Translate to bbox center
    corners += np.array([x, y, z])
    
    return corners


def create_bbox_cylinders(corners, color, radius=0.05):
    """
    Create cylinder meshes for all edges of a bounding box.
    corners: (8, 3) array of corner coordinates
    color: [r, g, b] color for the cylinders
    radius: radius of the cylinders
    Returns: list of cylinder meshes
    """
    # Define edges connecting corners (based on corner order above)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Top face (z = h/2)
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face (z = -h/2)
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    cylinders = []
    for start_idx, end_idx in edges:
        start = corners[start_idx]
        end = corners[end_idx]
        
        # Create cylinder between start and end
        height = np.linalg.norm(end - start)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        
        # Compute transformation to align cylinder
        # Default cylinder is along z-axis centered at origin
        direction = (end - start) / height
        z_axis = np.array([0, 0, 1])
        
        # Rotation axis and angle
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            
            # Create rotation matrix using Rodrigues' formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            # Parallel or anti-parallel
            if np.dot(z_axis, direction) > 0:
                R = np.eye(3)
            else:
                R = -np.eye(3)
                R[2, 2] = 1
        
        # Transform cylinder
        cylinder.rotate(R, center=[0, 0, 0])
        cylinder.translate((start + end) / 2)
        cylinder.paint_uniform_color(color)
        
        cylinders.append(cylinder)
    
    return cylinders


def visualize_point_cloud_with_bboxes(lidar_path, json_path, output_path="output.ply", create_output_dir=True):
    """
    Visualize point cloud with ground truth and predicted bounding boxes.
    
    Args:
        lidar_path: Path to lidar.bin file
        json_path: Path to prediction.json file
        output_path: Path to save the PLY file
        create_output_dir: Whether to create output directory if it doesn't exist
    
    Returns:
        dict: Dictionary containing IoU and other metrics from the prediction
    """
    # Create output directory if needed
    if create_output_dir:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
    
    print(f"Loading point cloud from: {lidar_path}")
    pcd_full, xyz, rgb = load_lidar_bin(lidar_path)
    print(f"Loaded {len(xyz)} points")
    
    print(f"Loading bboxes from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_bbox = data['gt_box'][0]  # Take first bbox from list
    pred_bbox = data['pred_box']
    iou = data['ious'][0] if 'ious' in data and len(data['ious']) > 0 else 0.0
    
    print(f"GT bbox (7D): {gt_bbox}")
    print(f"Pred bbox (6D): {pred_bbox}")
    print(f"IoU: {iou:.4f}")
    
    # Use extract_points_in_bbox_3d from box_util.py to find points inside GT bbox
    points_in_gt, mask_gt = extract_points_in_bbox_3d(pcd_full, gt_bbox)
    colors = rgb.copy()
    colors[mask_gt] = [1.0, 1.0, 0.0]  # Yellow for points inside GT bbox
    
    print(f"Points inside GT bbox: {np.sum(mask_gt)}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create GT bbox (red) - using same coordinate system as box_util.py
    gt_corners = get_bbox_corners(gt_bbox, has_yaw=True)
    gt_cylinders = create_bbox_cylinders(gt_corners, color=[1.0, 0.0, 0.0], radius=0.05)
    
    # Create pred bbox (green)
    pred_corners = get_bbox_corners(pred_bbox, has_yaw=False)
    pred_cylinders = create_bbox_cylinders(pred_corners, color=[0.0, 1.0, 0.0], radius=0.05)
    
    # Combine all geometries
    geometries = [pcd] + gt_cylinders + pred_cylinders
    
    # Visualize
    print("Displaying visualization... (Close window to continue)")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Bounding Boxes",
        width=1920,
        height=1080,
        left=50,
        top=50
    )
    
    # Save to PLY file
    print(f"Saving to: {output_path}")
    
    # Combine all bbox meshes into one
    combined_bbox_mesh = o3d.geometry.TriangleMesh()
    for cylinder in gt_cylinders + pred_cylinders:
        combined_bbox_mesh += cylinder
    
    # Method 1: Save point cloud only (for point cloud viewers)
    pcd_output = output_path.replace('.ply', '_points.ply')
    o3d.io.write_point_cloud(pcd_output, pcd)
    print(f"Saved point cloud to: {pcd_output}")
    
    # Method 2: Save bbox meshes only (shows as solid cylinders)
    bbox_output = output_path.replace('.ply', '_bboxes.ply')
    o3d.io.write_triangle_mesh(bbox_output, combined_bbox_mesh)
    print(f"Saved bbox meshes to: {bbox_output}")
    
    # Method 3: Create a combined mesh (point cloud + bbox cylinders as mesh)
    # Convert point cloud to small spheres or just keep as points, then add bbox mesh
    # Since we can't directly combine PointCloud and TriangleMesh in one file,
    # we create a mesh version where points are also represented as mesh vertices
    
    # Add point cloud as vertices to the combined mesh
    pcd_vertices = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    
    # Get bbox mesh vertices and faces
    bbox_vertices = np.asarray(combined_bbox_mesh.vertices)
    bbox_triangles = np.asarray(combined_bbox_mesh.triangles)
    bbox_colors = np.asarray(combined_bbox_mesh.vertex_colors)
    
    # Combine: point cloud vertices + bbox mesh vertices
    combined_vertices = np.vstack([pcd_vertices, bbox_vertices])
    combined_colors = np.vstack([pcd_colors, bbox_colors])
    
    # Adjust triangle indices to account for point cloud vertices
    adjusted_triangles = bbox_triangles + len(pcd_vertices)
    
    # Create final combined mesh
    final_mesh = o3d.geometry.TriangleMesh()
    final_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    final_mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)
    final_mesh.triangles = o3d.utility.Vector3iVector(adjusted_triangles)
    
    combined_output = output_path.replace('.ply', '_combined.ply')
    o3d.io.write_triangle_mesh(combined_output, final_mesh)
    print(f"Saved combined visualization (with bbox meshes) to: {combined_output}")
    
    print("Done!")
    
    return {'iou': iou, 'gt_bbox': gt_bbox, 'pred_bbox': pred_bbox}


def visualize_from_prediction(
    dataset, 
    sequence, 
    frame_id, 
    base_dir="code/3eed",
    prediction_subdir="logs/3eed/drone/10-19-23-05/predictions",
    output_dir="vis_debug"
):
    """
    Visualize point cloud with bboxes based on dataset, sequence, and frame_id.
    
    Args:
        dataset: Dataset name (e.g., 'drone', 'quad', 'waymo')
        sequence: Sequence name (e.g., 'Outdoor_Day_penno_plaza')
        frame_id: Frame ID (e.g., '000385')
        base_dir: Base directory of the project
        prediction_subdir: Subdirectory where predictions are stored
        output_dir: Output directory for visualization files
    
    Returns:
        dict: Dictionary containing IoU and other metrics from the prediction
    """
    # Construct paths based on prediction json path structure
    # json_path: logs/3eed/drone/10-19-23-05/predictions/drone/Outdoor_Day_penno_plaza/000385/prediction.json
    json_path = os.path.join(
        base_dir, 
        prediction_subdir, 
        dataset, 
        sequence, 
        frame_id, 
        "prediction.json"
    )
    
    # Corresponding lidar path from data directory
    # lidar_path: data/3eed_merge/drone/Outdoor_Day_penno_plaza/000385/lidar.bin
    # For waymo dataset, use .npy files instead of .bin
    lidar_filename = "lidar.npy" if dataset == "waymo" else "lidar.bin"
    lidar_path = os.path.join(
        base_dir,
        "data/3eed_merge",
        dataset,
        sequence,
        frame_id,
        lidar_filename
    )
    assert os.path.exists(json_path), f"JSON path not found: {json_path}"
    assert os.path.exists(lidar_path), f"Lidar path not found: {lidar_path}"
    
    # First read the json to get IoU for filename
    with open(json_path, 'r') as f:
        data = json.load(f)
    iou = data['ious'][0] if 'ious' in data and len(data['ious']) > 0 else 0.0
    
    # Output path with IoU in filename (2 decimal places)
    output_filename = f"{dataset}_{sequence}_{frame_id}_iou_{iou:.2f}_visualization.ply"
    output_path = os.path.join(base_dir, output_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Visualizing: {dataset}/{sequence}/{frame_id} (IoU: {iou:.2f})")
    print(f"{'='*80}")
    print(f"JSON path: {json_path}")
    print(f"Lidar path: {lidar_path}")
    print(f"Output path: {output_path}")
    print(f"{'='*80}\n")
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"ERROR: Prediction JSON not found: {json_path}")
        return None
    if not os.path.exists(lidar_path):
        print(f"ERROR: Lidar file not found: {lidar_path}")
        return None
    
    # Run visualization
    result = visualize_point_cloud_with_bboxes(lidar_path, json_path, output_path, create_output_dir=True)
    return result


def sample_and_visualize_predictions(
    num_samples,
    dataset=None,
    sequence=None,
    base_dir="code/3eed",
    prediction_subdir="logs/3eed/drone/10-19-23-05/predictions",
    output_dir="vis_debug",
    seed=None
):
    """
    Randomly sample frames from prediction directory and visualize them.
    
    Args:
        num_samples: Number of frames to sample
        dataset: Dataset name (e.g., 'drone', 'quad', 'waymo'). If None, sample from all datasets.
        sequence: Sequence name. If None, sample from all sequences.
        base_dir: Base directory of the project
        prediction_subdir: Subdirectory where predictions are stored
        output_dir: Output directory for visualization files
        seed: Random seed for reproducibility. If None, use random sampling.
    
    Returns:
        list: List of results from visualizations
    """
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Build prediction directory path
    pred_dir = os.path.join(base_dir, prediction_subdir)
    
    # Find all prediction.json files
    if dataset and sequence:
        # Specific dataset and sequence
        search_pattern = os.path.join(pred_dir, dataset, sequence, "*", "prediction.json")
    elif dataset:
        # Specific dataset, all sequences
        search_pattern = os.path.join(pred_dir, dataset, "*", "*", "prediction.json")
    elif sequence:
        # All datasets, specific sequence
        search_pattern = os.path.join(pred_dir, "*", sequence, "*", "prediction.json")
    else:
        # All datasets and sequences
        search_pattern = os.path.join(pred_dir, "*", "*", "*", "prediction.json")
    
    # Find all matching prediction files
    prediction_files = glob.glob(search_pattern)
    
    print(f"\n{'='*80}")
    print(f"Found {len(prediction_files)} prediction files")
    print(f"Sampling {min(num_samples, len(prediction_files))} frames...")
    print(f"{'='*80}\n")
    
    if len(prediction_files) == 0:
        print(f"ERROR: No prediction files found matching pattern: {search_pattern}")
        return []
    
    # Randomly sample
    num_to_sample = min(num_samples, len(prediction_files))
    sampled_files = random.sample(prediction_files, num_to_sample)
    
    # Extract dataset, sequence, frame_id from each path
    results = []
    for i, json_path in enumerate(sampled_files, 1):
        # Parse path to extract dataset, sequence, frame_id
        # Path format: .../predictions/{dataset}/{sequence}/{frame_id}/prediction.json
        parts = json_path.split(os.sep)
        frame_id = parts[-2]
        sequence_name = parts[-3]
        dataset_name = parts[-4]
        
        print(f"\n{'#'*80}")
        print(f"# Sample {i}/{num_to_sample}: {dataset_name}/{sequence_name}/{frame_id}")
        print(f"{'#'*80}")
        
        try:
            result = visualize_from_prediction(
                dataset=dataset_name,
                sequence=sequence_name,
                frame_id=frame_id,
                base_dir=base_dir,
                prediction_subdir=prediction_subdir,
                output_dir=output_dir
            )
            if result:
                result['dataset'] = dataset_name
                result['sequence'] = sequence_name
                result['frame_id'] = frame_id
                results.append(result)
        except Exception as e:
            print(f"ERROR: Failed to visualize {dataset_name}/{sequence_name}/{frame_id}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: Successfully visualized {len(results)}/{num_to_sample} frames")
    print(f"{'='*80}")
    
    if results:
        print("\nIoU Statistics:")
        ious = [r['iou'] for r in results]
        print(f"  Mean IoU: {np.mean(ious):.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU: {np.min(ious):.4f}")
        print(f"  Max IoU: {np.max(ious):.4f}")
        print(f"\nTop 3 by IoU:")
        sorted_results = sorted(results, key=lambda x: x['iou'], reverse=True)
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {r['dataset']}/{r['sequence']}/{r['frame_id']} - IoU: {r['iou']:.4f}")
    
    return results


if __name__ == "__main__":

    sample_and_visualize_predictions(
        base_dir=".",
        prediction_subdir="3eed_logs/Train_quad_drone_waymo_Val_quad_drone_waymo/wo_saf/predictions",
        output_dir="vis_results/drone",
        num_samples=5,
        seed=42  # For reproducibility
    )
    