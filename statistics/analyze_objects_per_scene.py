import numpy as np
import json
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

SEED_PATH = 'data/3eed/3eed_merge'
SPLIT_DIR = 'data/3eed/3eed_merge/splits'
PLATFORMS = ['waymo', 'drone', 'quad']


def load_split_files():
    """Load split files and create mapping from sequence to split"""
    split_mapping = {}  # {platform: {sequence_name: split}}
    
    for platform in PLATFORMS:
        split_mapping[platform] = {}
        
        for split in ['train', 'val']:
            split_file = os.path.join(SPLIT_DIR, f"{platform}_{split}.txt")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    sequence_list = [line.rstrip() for line in f]
                    for sequence in sequence_list:
                        split_mapping[platform][sequence] = split
    
    return split_mapping


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


def get_sequence_from_path(path, platform):
    """Extract sequence name from path based on platform"""
    # Example path: /path/to/waymo/sequence_name/frame_id/meta_info.json
    # We need to extract sequence_name
    parts = path.split('/')
    
    # Find platform index
    platform_idx = -1
    for i, part in enumerate(parts):
        if part == platform:
            platform_idx = i
            break
    
    if platform_idx >= 0 and platform_idx + 1 < len(parts):
        # The sequence name is right after the platform directory
        sequence_name = parts[platform_idx + 1]
        return sequence_name
    
    return None


def get_split_from_path(path, platform, split_mapping):
    """Get split (train/val) from path using split mapping"""
    sequence = get_sequence_from_path(path, platform)
    
    if sequence and platform in split_mapping and sequence in split_mapping[platform]:
        return split_mapping[platform][sequence]
    
    return 'unknown'


def get_num_objects(meta_info):
    """Get number of objects from meta_info"""
    try:
        num_objects = meta_info['ground_info'][0]['others_num'] + 1
    except:
        try:
            num_objects = len(meta_info['others']) + 1
        except:
            num_objects = 1  # At least the target object
    return num_objects


def analyze_objects_per_scene():
    """Analyze number of objects per scene across platforms and splits"""
    
    print("Loading split files...")
    split_mapping = load_split_files()
    
    # Print split statistics
    for platform in PLATFORMS:
        train_count = sum(1 for s in split_mapping[platform].values() if s == 'train')
        val_count = sum(1 for s in split_mapping[platform].values() if s == 'val')
        print(f"  {platform}: {train_count} train sequences, {val_count} val sequences")
    
    print("\nFinding all meta_info.json files...")
    meta_files = find_all_meta_info_files(SEED_PATH)
    print(f"Found {len(meta_files)} meta_info.json files")
    
    # Statistics storage: platform -> split -> list of object counts
    stats = defaultdict(lambda: defaultdict(list))
    
    # Overall statistics
    platform_counts = defaultdict(int)
    split_counts = defaultdict(int)
    
    for meta_file in tqdm(meta_files, desc="Processing meta_info files"):
        try:
            # Get platform and split
            platform = get_platform_from_path(meta_file)
            if platform is None:
                continue
            
            split = get_split_from_path(meta_file, platform, split_mapping)
            
            # Load meta_info
            with open(meta_file, 'r') as f:
                meta_info = json.load(f)
            
            # Get number of objects
            num_objects = get_num_objects(meta_info)
            
            # Store statistics
            stats[platform][split].append(num_objects)
            stats[platform]['all'].append(num_objects)
            stats['all'][split].append(num_objects)
            stats['all']['all'].append(num_objects)
            
            platform_counts[platform] += 1
            split_counts[split] += 1
            
        except Exception as e:
            print(f"Error processing {meta_file}: {e}")
            continue
    
    # Print summary statistics
    print("\n" + "="*100)
    print("NUMBER OF OBJECTS PER SCENE - SUMMARY STATISTICS")
    print("="*100)
    
    print(f"\n{'Platform':<15} {'Split':<10} {'Scenes':<10} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Std':<10}")
    print("-" * 100)
    
    # Print statistics for each platform and split
    for platform in ['waymo', 'drone', 'quad', 'all']:
        platform_name = platform.capitalize() if platform != 'waymo' else 'Vehicle' if platform != 'all' else 'ALL'
        
        for split in ['train', 'val', 'test', 'all']:
            if len(stats[platform][split]) > 0:
                obj_counts = stats[platform][split]
                print(f"{platform_name:<15} {split:<10} {len(obj_counts):<10} "
                      f"{np.mean(obj_counts):<10.2f} {np.median(obj_counts):<10.0f} "
                      f"{np.min(obj_counts):<10} {np.max(obj_counts):<10} "
                      f"{np.std(obj_counts):<10.2f}")
        print("-" * 100)
    
    # Print distribution statistics in table format
    print("\n" + "="*100)
    print("Table: Scene count grouped by number of objects per scene across platforms and splits")
    print("="*100)
    
    # Define bins for object count ranges
    bin_ranges = [(1, 3), (4, 6), (7, 9), (10, 12), (13, float('inf'))]
    bin_labels = ['1-3', '4-6', '7-9', '10-12', '13+']
    
    # Print header
    print(f"\n{'Platform':<20} {'1-3':<12} {'4-6':<12} {'7-9':<12} {'10-12':<12} {'13+':<12} {'Total':<12}")
    print("="*110)
    
    # Training section
    print("\nTraining")
    train_totals = {label: 0 for label in bin_labels}
    train_grand_total = 0
    
    for platform in ['waymo', 'drone', 'quad']:
        platform_name = '🚗 Vehicle' if platform == 'waymo' else '🛸 Drone' if platform == 'drone' else '🐕 Quadruped'
        
        if len(stats[platform]['train']) > 0:
            obj_counts = np.array(stats[platform]['train'])
            counts = []
            
            for min_obj, max_obj in bin_ranges:
                if max_obj == float('inf'):
                    count = np.sum(obj_counts >= min_obj)
                else:
                    count = np.sum((obj_counts >= min_obj) & (obj_counts <= max_obj))
                counts.append(count)
            
            total = sum(counts)
            print(f"{platform_name:<20} {counts[0]:<12,} {counts[1]:<12,} {counts[2]:<12,} {counts[3]:<12,} {counts[4]:<12,} {total:<12,}")
            
            # Add to totals
            for i, label in enumerate(bin_labels):
                train_totals[label] += counts[i]
            train_grand_total += total
    
    # Print training total
    print(f"{'Total':<20} {train_totals['1-3']:<12,} {train_totals['4-6']:<12,} {train_totals['7-9']:<12,} {train_totals['10-12']:<12,} {train_totals['13+']:<12,} {train_grand_total:<12,}")
    
    # Validation section
    print("\nValidation")
    val_totals = {label: 0 for label in bin_labels}
    val_grand_total = 0
    
    for platform in ['waymo', 'drone', 'quad']:
        platform_name = '🚗 Vehicle' if platform == 'waymo' else '🛸 Drone' if platform == 'drone' else '🐕 Quadruped'
        
        if len(stats[platform]['val']) > 0:
            obj_counts = np.array(stats[platform]['val'])
            counts = []
            
            for min_obj, max_obj in bin_ranges:
                if max_obj == float('inf'):
                    count = np.sum(obj_counts >= min_obj)
                else:
                    count = np.sum((obj_counts >= min_obj) & (obj_counts <= max_obj))
                counts.append(count)
            
            total = sum(counts)
            print(f"{platform_name:<20} {counts[0]:<12,} {counts[1]:<12,} {counts[2]:<12,} {counts[3]:<12,} {counts[4]:<12,} {total:<12,}")
            
            # Add to totals
            for i, label in enumerate(bin_labels):
                val_totals[label] += counts[i]
            val_grand_total += total
    
    # Print validation total
    print(f"{'Total':<20} {val_totals['1-3']:<12,} {val_totals['4-6']:<12,} {val_totals['7-9']:<12,} {val_totals['10-12']:<12,} {val_totals['13+']:<12,} {val_grand_total:<12,}")
    
    # Summary section
    print("\n" + "="*110)
    summary_totals = {label: train_totals[label] + val_totals[label] for label in bin_labels}
    summary_grand_total = train_grand_total + val_grand_total
    print(f"{'Summary':<20} {summary_totals['1-3']:<12,} {summary_totals['4-6']:<12,} {summary_totals['7-9']:<12,} {summary_totals['10-12']:<12,} {summary_totals['13+']:<12,} {summary_grand_total:<12,}")
    print("="*110)
    
    # Create visualization
    print("\n" + "="*100)
    print("CREATING VISUALIZATIONS...")
    print("="*100)
    
    # Prepare data for plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Number of Objects per Scene - Distribution', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram for each platform (all splits combined)
    ax1 = axes[0, 0]
    for platform in ['waymo', 'drone', 'quad']:
        platform_name = platform.capitalize() if platform != 'waymo' else 'Vehicle'
        if len(stats[platform]['all']) > 0:
            ax1.hist(stats[platform]['all'], bins=range(1, 21), alpha=0.6, label=platform_name, edgecolor='black')
    ax1.set_xlabel('Number of Objects per Scene', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution by Platform (All Splits)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by platform
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    for platform in ['waymo', 'drone', 'quad']:
        if len(stats[platform]['all']) > 0:
            box_data.append(stats[platform]['all'])
            box_labels.append(platform.capitalize() if platform != 'waymo' else 'Vehicle')
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['skyblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Number of Objects per Scene', fontsize=12)
    ax2.set_title('Distribution by Platform (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Bar chart showing mean objects per scene
    ax3 = axes[1, 0]
    platforms_to_plot = ['waymo', 'drone', 'quad']
    means = [np.mean(stats[p]['all']) if len(stats[p]['all']) > 0 else 0 for p in platforms_to_plot]
    labels = [p.capitalize() if p != 'waymo' else 'Vehicle' for p in platforms_to_plot]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = ax3.bar(labels, means, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Mean Objects per Scene', fontsize=12)
    ax3.set_title('Average Objects per Scene by Platform', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Cumulative distribution
    ax4 = axes[1, 1]
    for platform in ['waymo', 'drone', 'quad']:
        platform_name = platform.capitalize() if platform != 'waymo' else 'Vehicle'
        if len(stats[platform]['all']) > 0:
            sorted_data = np.sort(stats[platform]['all'])
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            ax4.plot(sorted_data, cumulative, label=platform_name, linewidth=2)
    
    ax4.set_xlabel('Number of Objects per Scene', fontsize=12)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax4.set_title('Cumulative Distribution by Platform', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'code/3eed/objects_per_scene_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Save detailed statistics to JSON
    output_data = {}
    for platform in PLATFORMS + ['all']:
        output_data[platform] = {}
        for split in ['train', 'val', 'test', 'all']:
            if len(stats[platform][split]) > 0:
                obj_counts = stats[platform][split]
                output_data[platform][split] = {
                    'num_scenes': len(obj_counts),
                    'mean': float(np.mean(obj_counts)),
                    'median': float(np.median(obj_counts)),
                    'min': int(np.min(obj_counts)),
                    'max': int(np.max(obj_counts)),
                    'std': float(np.std(obj_counts)),
                    'percentile_25': float(np.percentile(obj_counts, 25)),
                    'percentile_75': float(np.percentile(obj_counts, 75)),
                }
    
    json_output_path = 'code/3eed/objects_per_scene_statistics.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Statistics saved to: {json_output_path}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    analyze_objects_per_scene()

