#!/usr/bin/env python3
"""
Analyze 3eed_merge dataset statistics.
Count JSON files and objects (ground_info) for each platform (quad, drone, waymo) and their train/val splits.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def read_split_file(split_file):
    """Read split file and return list of scene names."""
    if not os.path.exists(split_file):
        return []
    with open(split_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes

def count_json_and_objects_in_scene(scene_path):
    """Count all JSON files and objects (ground_info) in a scene directory.
    
    Returns:
        json_count: total number of JSON files
        object_count: total number of objects
        objects_per_json: list of object counts for each JSON file
    """
    json_count = 0
    object_count = 0
    objects_per_json = []
    
    for root, dirs, files in os.walk(scene_path):
        for file in files:
            if file.endswith('.json'):
                json_count += 1
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'ground_info' in data and isinstance(data['ground_info'], list):
                            obj_count = 1 + len(data['ground_info'][0]['others'])
                            object_count += obj_count
                            objects_per_json.append(obj_count)
                            # ! Whether to count 'others'?
                except Exception as e:
                    print(f"  Warning: Failed to read {json_path}: {e}")
    
    return json_count, object_count, objects_per_json

def count_objects_distribution(objects_per_json_list, platform_name):
    """Count distribution of JSON files by object count ranges based on platform.
    
    Args:
        objects_per_json_list: list of object counts for each JSON file
        platform_name: 'waymo', 'drone', or 'quad'
    
    Returns:
        dict: distribution of JSON files by object count ranges
    """
    distribution = defaultdict(int)
    
    if platform_name == 'waymo':
        # Waymo: <5, 5-10, 10-15, 15-20, >20
        ranges = [
            ('0-4', 0, 5),
            ('5-9', 5, 10),
            ('10-14', 10, 15),
            ('15-19', 15, 20),
            ('20+', 20, float('inf'))
        ]
    elif platform_name == 'drone':
        # Drone: 0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 12-14, 14-16, >16
        ranges = [
            ('0-1', 0, 2),
            ('2-3', 2, 4),
            ('4-5', 4, 6),
            ('6-7', 6, 8),
            ('8-9', 8, 10),
            ('10-11', 10, 12),
            ('12-13', 12, 14),
            ('14-15', 14, 16),
            ('16+', 16, float('inf'))
        ]
    elif platform_name == 'quad':
        # Quad: 0-2, 3, 4, >4
        ranges = [
            ('0-2', 0, 3),
            ('3', 3, 4),
            ('4', 4, 5),
            ('5+', 5, float('inf'))
        ]
    else:
        raise ValueError(f"Unknown platform: {platform_name}")
    
    # Count for each range
    for obj_count in objects_per_json_list:
        for label, min_val, max_val in ranges:
            if min_val <= obj_count < max_val:
                distribution[label] += 1
                break
    
    return dict(distribution), ranges

def analyze_platform(data_root, splits_dir, platform_name):
    """Analyze a single platform (quad/drone/waymo)."""
    print(f"\n{'='*60}")
    print(f"Platform: {platform_name.upper()}")
    print(f"{'='*60}")
    
    platform_dir = os.path.join(data_root, platform_name)
    
    # Read train/val splits
    train_file = os.path.join(splits_dir, f'{platform_name}_train.txt')
    val_file = os.path.join(splits_dir, f'{platform_name}_val.txt')
    assert os.path.exists(train_file), f"train_file not found: {train_file}"
    assert os.path.exists(val_file), f"val_file not found: {val_file}"
    
    train_scenes = read_split_file(train_file)
    val_scenes = read_split_file(val_file)
    
    print(f"\nTrain scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")
    
    # Count JSON files and objects
    train_json_count = 0
    val_json_count = 0
    total_json_count = 0
    
    train_object_count = 0
    val_object_count = 0
    total_object_count = 0
    
    train_scene_details = []
    val_scene_details = []
    
    # For object distribution analysis
    all_objects_per_json = []
    train_objects_per_json = []
    val_objects_per_json = []
    
    # Check if platform directory exists
    if not os.path.exists(platform_dir):
        print(f"Warning: Platform directory not found: {platform_dir}")
        return
    
    # Get all scenes in platform directory
    all_scenes = []
    for item in os.listdir(platform_dir):
        item_path = os.path.join(platform_dir, item)
        if os.path.isdir(item_path):
            all_scenes.append(item)
    
    print(f"\nTotal scenes found in directory: {len(all_scenes)}")
    
    # Count JSON files and objects for each scene
    for scene in all_scenes:
        scene_path = os.path.join(platform_dir, scene)
        json_count, object_count, objects_per_json = count_json_and_objects_in_scene(scene_path)
        total_json_count += json_count
        total_object_count += object_count
        all_objects_per_json.extend(objects_per_json)
        
        if scene in train_scenes:
            train_json_count += json_count
            train_object_count += object_count
            train_scene_details.append((scene, json_count, object_count))
            train_objects_per_json.extend(objects_per_json)
        elif scene in val_scenes:
            val_json_count += json_count
            val_object_count += object_count
            val_scene_details.append((scene, json_count, object_count))
            val_objects_per_json.extend(objects_per_json)
        else:
            print(f"  Warning: Scene '{scene}' not in train or val split")
    
    # Print summary
    print(f"\n{'-'*60}")
    print(f"Statistics:")
    print(f"{'-'*60}")
    print(f"Total JSON files: {total_json_count}")
    print(f"Train JSON files: {train_json_count} ({train_json_count/total_json_count*100:.1f}%)" if total_json_count > 0 else "Train JSON files: 0")
    print(f"Val JSON files: {val_json_count} ({val_json_count/total_json_count*100:.1f}%)" if total_json_count > 0 else "Val JSON files: 0")
    
    print(f"\nTotal Objects: {total_object_count}")
    print(f"Train Objects: {train_object_count} ({train_object_count/total_object_count*100:.1f}%)" if total_object_count > 0 else "Train Objects: 0")
    print(f"Val Objects: {val_object_count} ({val_object_count/total_object_count*100:.1f}%)" if total_object_count > 0 else "Val Objects: 0")
    
    # Print object distribution by ranges (Train and Val separately)
    print(f"\n{'-'*60}")
    print(f"Object Distribution (JSON files by object count):")
    print(f"{'-'*60}")
    
    if train_objects_per_json or val_objects_per_json:
        _, ranges = count_objects_distribution(all_objects_per_json if all_objects_per_json else [], platform_name)
        
        # Train distribution
        if train_objects_per_json:
            train_dist, _ = count_objects_distribution(train_objects_per_json, platform_name)
            print(f"\nTrain Split:")
            for label, _, _ in ranges:
                count = train_dist.get(label, 0)
                pct = count / train_json_count * 100 if train_json_count > 0 else 0
                print(f"  {label:>8} objects: {count:>6} JSON files ({pct:>5.1f}%)")
        
        # Val distribution
        if val_objects_per_json:
            val_dist, _ = count_objects_distribution(val_objects_per_json, platform_name)
            print(f"\nVal Split:")
            for label, _, _ in ranges:
                count = val_dist.get(label, 0)
                pct = count / val_json_count * 100 if val_json_count > 0 else 0
                print(f"  {label:>8} objects: {count:>6} JSON files ({pct:>5.1f}%)")
    
    # Print detailed scene counts (top 5 from each split)
    # if train_scene_details:
    #     print(f"\n{'-'*60}")
    #     print(f"Top 5 Train scenes by object count:")
    #     train_scene_details.sort(key=lambda x: x[2], reverse=True)
    #     for scene, json_cnt, obj_cnt in train_scene_details[:5]:
    #         print(f"  {scene}: {json_cnt} JSON files, {obj_cnt} objects")
    
    # if val_scene_details:
    #     print(f"\nTop 5 Val scenes by object count:")
    #     val_scene_details.sort(key=lambda x: x[2], reverse=True)
    #     for scene, json_cnt, obj_cnt in val_scene_details[:5]:
    #         print(f"  {scene}: {json_cnt} JSON files, {obj_cnt} objects")
    
    return {
        'platform': platform_name,
        'total_scenes': len(all_scenes),
        'train_scenes': len(train_scenes),
        'val_scenes': len(val_scenes),
        'total_json': total_json_count,
        'train_json': train_json_count,
        'val_json': val_json_count,
        'total_objects': total_object_count,
        'train_objects': train_object_count,
        'val_objects': val_object_count
    }

def main():
    # Dataset root directory
    data_root = 'code/3eed/data/3eed_merge'
    # data_root = 'data/3eed'
    splits_dir = 'code/3eed/data/splits'
    # Alternative: use relative paths if running from project root
    # data_root = 'data/3eed_merge'
    # splits_dir = 'data/3eed_merge/splits'
    
    assert os.path.exists(data_root), f"data_root not found: {data_root}"
    assert os.path.exists(splits_dir), f"splits_dir not found: {splits_dir}"
    
    print("=" * 60)
    print("3EED MERGE DATASET ANALYSIS")
    print("=" * 60)
    print(f"Dataset path: {data_root}")
    
    platforms = ['waymo', 'drone', 'quad']
    results = []
    
    for platform in platforms:
        result = analyze_platform(data_root, splits_dir, platform)
        if result:
            results.append(result)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Platform':<10} {'Scenes':<8} {'Train/Val':<12} {'Total JSON':<12} {'Train JSON':<12} {'Val JSON':<12} {'Total Objs':<12} {'Train Objs':<12} {'Val Objs':<12}")
    print("-" * 120)
    
    total_scenes = 0
    total_json = 0
    total_train_json = 0
    total_val_json = 0
    total_objects = 0
    total_train_objects = 0
    total_val_objects = 0
    
    for result in results:
        total_scenes += result['total_scenes']
        total_json += result['total_json']
        total_train_json += result['train_json']
        total_val_json += result['val_json']
        total_objects += result['total_objects']
        total_train_objects += result['train_objects']
        total_val_objects += result['val_objects']
        
        print(f"{result['platform']:<10} {result['total_scenes']:<8} "
              f"{result['train_scenes']}/{result['val_scenes']:<10} "
              f"{result['total_json']:<12} {result['train_json']:<12} {result['val_json']:<12} "
              f"{result['total_objects']:<12} {result['train_objects']:<12} {result['val_objects']:<12}")
    
    print("-" * 120)
    print(f"{'TOTAL':<10} {total_scenes:<8} {'':<12} "
          f"{total_json:<12} {total_train_json:<12} {total_val_json:<12} "
          f"{total_objects:<12} {total_train_objects:<12} {total_val_objects:<12}")
    
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()