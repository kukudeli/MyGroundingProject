import numpy as np
import json
import os
from tqdm import tqdm
from collections import defaultdict
import sys
import random

# Add the project root to the path
sys.path.insert(0, 'code/3eed')

# Import the extract_points_in_bbox_3d function
from src.joint_det_dataset import extract_points_in_bbox_3d

SEED_PATH = 'data/3eed/3eed_merge'
PLATFORMS = ['waymo', 'drone', 'quad']


def find_all_meta_info_files(root_dir):
    """Find all meta_info.json files in the dataset"""
    meta_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'meta_info.json':
                full_path = os.path.join(dirpath, filename)
                meta_files.append(full_path)
    return meta_files


def get_platform_from_path(path):
    """Detect platform from path"""
    if '/drone/' in path.lower():
        return 'drone'
    elif '/quad/' in path.lower():
        return 'quad'
    elif '/waymo/' in path.lower():
        return 'waymo'
    else:
        return None


def get_num_objects(meta_info):
    """Get number of objects from meta_info"""
    try:
        num_objects = meta_info['ground_info'][0]['others_num'] + 1
    except:
        num_objects = len(meta_info['others']) + 1
    return num_objects


def load_point_cloud(pcd_path):
    """Load point cloud from file"""
    if pcd_path.endswith(".bin"):
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    else:
        pcd = np.load(pcd_path)
    return pcd


def get_points_in_bbox(xyz, bbox, dataset):
    """Get points inside bbox for a specific dataset type"""
    
    points_in_bbox, point_mask = extract_points_in_bbox_3d(xyz, bbox[:7])
    return points_in_bbox, point_mask


def analyze_dataset():
    """Analyze dataset statistics for all platforms"""
    
    print("Finding all meta_info.json files...")
    meta_files = find_all_meta_info_files(SEED_PATH)
    print(f"Found {len(meta_files)} meta_info.json files")
    
    # Statistics storage
    platform_stats = {
        'waymo': {
            'num_scenes': 0,
            'total_objects': 0,
            'total_points_per_object': [],
            'objects_per_scene': [],
        },
        'drone': {
            'num_scenes': 0,
            'total_objects': 0,
            'total_points_per_object': [],
            'objects_per_scene': [],
        },
        'quad': {
            'num_scenes': 0,
            'total_objects': 0,
            'total_points_per_object': [],
            'objects_per_scene': [],
        },
    }
    
    skipped_files = []
    
    # meta_files = random.sample(meta_files, 200)
    for meta_file in tqdm(meta_files, desc="Processing meta_info files"):
        try:
            # Get platform
            platform = get_platform_from_path(meta_file)
            if platform is None:
                skipped_files.append(meta_file)
                continue
            
            # Load meta_info
            with open(meta_file, 'r') as f:
                meta_info = json.load(f)
            
            # Get scene directory
            scene_dir = os.path.dirname(meta_file)
            
            # Load point cloud - try different file names
            pcd_path = None
            for filename in ['point_cloud.bin', 'lidar.bin', 'point_cloud.npy', 'lidar.npy']:
                test_path = os.path.join(scene_dir, filename)
                if os.path.exists(test_path):
                    pcd_path = test_path
                    break
            
            if pcd_path is None:
                # print(f"Warning: Point cloud not found for {meta_file}")
                continue
            
            # Get number of objects (only count if we have point cloud)
            num_objects = get_num_objects(meta_info)
            platform_stats[platform]['num_scenes'] += 1
            platform_stats[platform]['objects_per_scene'].append(num_objects)
            
            pcd = load_point_cloud(pcd_path)
            xyz = pcd[:, :3]
            
            # Process target object and others from ground_info structure
            if 'ground_info' in meta_info and len(meta_info['ground_info']) > 0:
                ground_info = meta_info['ground_info'][0]
                
                # Process target object
                if 'bbox_3d' in ground_info:
                    gt_bbox = np.array(ground_info['bbox_3d'])
                    points_in_bbox, _ = get_points_in_bbox(xyz, gt_bbox, platform)
                    num_points = len(points_in_bbox)
                    platform_stats[platform]['total_points_per_object'].append(num_points)
                    platform_stats[platform]['total_objects'] += 1
                
                # Process other objects
                if 'others' in ground_info:
                    for other_obj in ground_info['others']:
                        if 'bbox_3d_other' in other_obj:
                            other_bbox = np.array(other_obj['bbox_3d_other'])
                            points_in_bbox, _ = get_points_in_bbox(xyz, other_bbox, platform)
                            num_points = len(points_in_bbox)
                            platform_stats[platform]['total_points_per_object'].append(num_points)
                            platform_stats[platform]['total_objects'] += 1
            
            # Fallback: try old structure (gt_bbox, others with bbox)
            elif 'gt_bbox' in meta_info:
                gt_bbox = np.array(meta_info['gt_bbox'])
                points_in_bbox, _ = get_points_in_bbox(xyz, gt_bbox, platform)
                num_points = len(points_in_bbox)
                platform_stats[platform]['total_points_per_object'].append(num_points)
                platform_stats[platform]['total_objects'] += 1
                
                if 'others' in meta_info:
                    for other_obj in meta_info['others']:
                        if 'bbox' in other_obj:
                            other_bbox = np.array(other_obj['bbox'])
                            points_in_bbox, _ = get_points_in_bbox(xyz, other_bbox, platform)
                            num_points = len(points_in_bbox)
                            platform_stats[platform]['total_points_per_object'].append(num_points)
                            platform_stats[platform]['total_objects'] += 1
            
        except Exception as e:
            print(f"Error processing {meta_file}: {e}")
            skipped_files.append(meta_file)
            continue
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\n{'Platform':<15} {'Num Scenes':<15} {'Avg Objects/Scene':<20} {'Avg Points/Object':<20}")
    print("-" * 80)
    
    for platform in PLATFORMS:
        stats = platform_stats[platform]
        num_scenes = stats['num_scenes']
        
        if num_scenes > 0:
            avg_objects_per_scene = np.mean(stats['objects_per_scene'])
        else:
            avg_objects_per_scene = 0
        
        if len(stats['total_points_per_object']) > 0:
            avg_points_per_object = np.mean(stats['total_points_per_object'])
        else:
            avg_points_per_object = 0
        
        platform_name = platform.capitalize() if platform != 'waymo' else 'Vehicle'
        print(f"{platform_name:<15} {num_scenes:<15} {avg_objects_per_scene:<20.2f} {avg_points_per_object:<20.2f}")
    
    # Detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for platform in PLATFORMS:
        stats = platform_stats[platform]
        platform_name = platform.capitalize() if platform != 'waymo' else 'Vehicle'
        
        print(f"\n{platform_name}:")
        print(f"  Total scenes: {stats['num_scenes']}")
        print(f"  Total objects: {stats['total_objects']}")
        
        print(f"  Objects per scene:")
        print(f"    Mean: {np.mean(stats['objects_per_scene']):.2f}")
        print(f"    Median: {np.median(stats['objects_per_scene']):.2f}")
        print(f"    Min: {np.min(stats['objects_per_scene'])}")
        print(f"    Max: {np.max(stats['objects_per_scene'])}")
        print(f"    Std: {np.std(stats['objects_per_scene']):.2f}")
        
        # if len(stats['total_points_per_object']) > 0:
        print(f"  Points per object:")
        print(f"    Mean: {np.mean(stats['total_points_per_object']):.2f}")
        print(f"    Median: {np.median(stats['total_points_per_object']):.2f}")
        print(f"    Min: {np.min(stats['total_points_per_object'])}")
        print(f"    Max: {np.max(stats['total_points_per_object'])}")
        print(f"    Std: {np.std(stats['total_points_per_object']):.2f}")
    
    if skipped_files:
        print(f"\n\nSkipped {len(skipped_files)} files due to errors")
    
    # Save statistics to JSON
    output_file = os.path.join(SEED_PATH, 'dataset_statistics.json')
    output_data = {}
    for platform in PLATFORMS:
        stats = platform_stats[platform]
        output_data[platform] = {
            'num_scenes': int(stats['num_scenes']),
            'total_objects': int(stats['total_objects']),
            'avg_objects_per_scene': float(np.mean(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
            'avg_points_per_object': float(np.mean(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
            'objects_per_scene_stats': {
                'mean': float(np.mean(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
                'median': float(np.median(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
                'min': int(np.min(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
                'max': int(np.max(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
                'std': float(np.std(stats['objects_per_scene'])) if len(stats['objects_per_scene']) > 0 else 0,
            },
            'points_per_object_stats': {
                'mean': float(np.mean(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
                'median': float(np.median(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
                'min': int(np.min(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
                'max': int(np.max(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
                'std': float(np.std(stats['total_points_per_object'])) if len(stats['total_points_per_object']) > 0 else 0,
            }
        }
    
    # with open(output_file, 'w', encoding='utf-8') as f:
        # json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    # print(f"\n\nStatistics saved to: {output_file}")


if __name__ == "__main__":
    analyze_dataset()

