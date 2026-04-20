"""
Compute statistics:
- Across platforms (Vehicle/Waymo, Drone, Quadruped)
- Across different object-count ranges (1-3, 4-6, 7-9, >9)
- Accuracy metrics (Acc@25, Acc@50)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys
import os
from tqdm import tqdm 
from collections import defaultdict
import json
import ipdb

SEED_PATH = 'data/3eed/3eed_merge'
PLATFORMS = ['waymo', 'drone', 'quad']


def find_bin_files(root_dir, end_file='.json'):
    assert os.path.exists(root_dir), f"Root directory does not exist: {root_dir}"
    # ipdb.set_trace()
    bin_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # print(f"dirpath: {dirpath}")
        # print(f"dirnames: {dirnames}")
        # print(f"filenames: {filenames}")
        if any(platform in dirpath for platform in PLATFORMS):
            for filename in filenames:
                if filename.endswith(end_file):
                    full_path = os.path.join(dirpath, filename)
                    bin_files.append(full_path)
    return bin_files

class TEED_Metric:
    '''
    TEED Metric
    '''
    def __init__(self, threshold=[0.25, 0.50]):
        self.threshold = threshold
        self.platform_total = defaultdict(int)
        self.platform_iou_scores = defaultdict(list)
        self.num_objects_total = defaultdict(int)
        self.num_objects_iou_scores = defaultdict(list)
        
        # Store detailed data for object count ranges
        self.platform_object_count_data = defaultdict(lambda: defaultdict(list))  # platform -> object_count -> [ious]
        self.platform_object_count_correct = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # platform -> object_count -> threshold -> count

        for thr in threshold:
            setattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}", defaultdict(int))
            setattr(self, f"number_boxes_acc{'{:02.0f}'.format(thr * 100)}", defaultdict(int))

    def display_mertic(self):
        # miou
        miou_result = {}
        all_ious = []
        all_numbers = 0
        for class_name in PLATFORMS:
            iou_value = self.platform_iou_scores[class_name]
            class_number = self.platform_total[class_name]
            all_ious.extend(iou_value)
            all_numbers += class_number
            if class_number != 0:
                miou_result.update({f'{class_name}': "{:.2f}".format(100*sum(iou_value) / class_number)})
            else:
                miou_result.update({f'{class_name}': "{:.2f}".format(0)})
        miou_result.update({'mIoU': "{:.2f}".format(100*sum(all_ious) / all_numbers)})
        # acc
        acc_result = {}
        for thr in self.threshold:
            thr_all_numbers = 0
            thr_miou_result = {}
            for class_name in PLATFORMS:
                class_number = getattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}")[class_name]
                thr_all_numbers += class_number
                if class_number != 0:
                    thr_miou_result.update({f'{class_name}': "{:.2f}".format(100*class_number / self.platform_total[class_name])})
                else:
                    thr_miou_result.update({f'{class_name}': "{:.2f}".format(0)})

            thr_miou_result.update({'ALL': "{:.2f}".format(100*thr_all_numbers / all_numbers)})
            acc_result.update({f"thr_acc{'{:02.0f}'.format(thr * 100)}": thr_miou_result})

        print("======================IoU======================")
        for key ,value in miou_result.items():
            print(f"{key}:{value}")
        print("======================Acc======================")
        for key ,value in acc_result.items():
            print(f"{key}:{value}")
        return miou_result, acc_result

    def display_object_count_table(self):
        """Display results in the format shown in the image - by object count ranges"""
        # Define object count ranges
        ranges = [
            (1, 3, "1 – 3"),
            (4, 6, "4 – 6"), 
            (7, 9, "7 – 9"),
            (10, float('inf'), "> 9")
        ]
        
        print("\n======================The number of objects per scene======================")
        print(f"{'Object Count':<15} {'Vehicle':<20} {'Drone':<20} {'Quadruped':<20}")
        print(f"{'':<15} {'Acc@25':<10} {'Acc@50':<10} {'Acc@25':<10} {'Acc@50':<10} {'Acc@25':<10} {'Acc@50':<10}")
        print("-" * 95)
        
        for min_obj, max_obj, range_str in ranges:
            # Calculate accuracy for this range
            range_acc25 = {}
            range_acc50 = {}
            
            for platform in PLATFORMS:
                # Count objects in this range for this platform
                range_total = 0
                range_correct25 = 0
                range_correct50 = 0
                
                # Go through all object counts for this platform
                for obj_count in self.platform_object_count_data[platform]:
                    if min_obj <= obj_count <= max_obj:
                        # Count total samples in this range
                        range_total += len(self.platform_object_count_data[platform][obj_count])
                        
                        # Count correct samples for each threshold
                        range_correct25 += self.platform_object_count_correct[platform][obj_count].get(0.25, 0)
                        range_correct50 += self.platform_object_count_correct[platform][obj_count].get(0.50, 0)
                
                if range_total > 0:
                    range_acc25[platform] = 100.0 * range_correct25 / range_total
                    range_acc50[platform] = 100.0 * range_correct50 / range_total
                else:
                    range_acc25[platform] = 0.0
                    range_acc50[platform] = 0.0
            
            # Print the row
            print(f"{range_str:<15} {range_acc25['waymo']:<10.2f} {range_acc50['waymo']:<10.2f} {range_acc25['drone']:<10.2f} {range_acc50['drone']:<10.2f} {range_acc25['quad']:<10.2f} {range_acc50['quad']:<10.2f}")
        
        return ranges

    def record_single(self, rel_dict):
        platform = rel_dict['platform']
        num_objects = rel_dict['num_objects']
        iou = rel_dict['iou']

        self.platform_total[platform] += 1
        self.num_objects_total[num_objects] += 1
        self.platform_iou_scores[platform].append(iou)
        self.num_objects_iou_scores[num_objects].append(iou)
        
        # Store detailed data for object count ranges
        self.platform_object_count_data[platform][num_objects].append(iou)

        for thr in self.threshold:
            if iou >= thr:
                recoder = getattr(self, f"class_acc{'{:02.0f}'.format(thr * 100)}")
                recoder[platform] += 1
                recoder = getattr(self, f"number_boxes_acc{'{:02.0f}'.format(thr * 100)}")
                recoder[num_objects] += 1
                
                # Store detailed correct counts
                self.platform_object_count_correct[platform][num_objects][thr] += 1


class Tester:
    def __init__(self, prediction_path, object_num_split = [3, 6, 9, 10]):
        self.prediction_path = prediction_path
        self.object_num_split = object_num_split
        self.metric_recoder = TEED_Metric()
        for i in object_num_split:
            setattr(self, f'metric_recoder_{str(i).zfill(2)}', TEED_Metric())

    def get_platform(self, prediction_json_path: str):
        # Check for specific platform patterns in the path
        if '/drone/' in prediction_json_path:
            return 'drone'
        elif '/quad/' in prediction_json_path:
            return 'quad'
        elif '/waymo/' in prediction_json_path:
            return 'waymo'
        else:
            raise ValueError(f"Unknown platform in {prediction_json_path}")

    def get_meta_info(self, meta_info_path):
        with open(meta_info_path, 'rb') as f:
            meta_info = json.load(f)
        try:
            num_objects = meta_info['ground_info'][0]['others_num'] + 1
        except:
            num_objects = len(meta_info['others']) + 1

        return num_objects

    def get_record_name(self, num_objects):
        if num_objects <= self.object_num_split[0]:
            return 'metric_recoder_03'
        elif num_objects <= self.object_num_split[1]:
            return 'metric_recoder_06'
        elif num_objects <= self.object_num_split[2]:
            return 'metric_recoder_09'
        else:
            return 'metric_recoder_10'

    def calculate_metric(self):
        results_list = []
        # extrac predictions
        prediction_jsons = find_bin_files(self.prediction_path, end_file='.json')
        
        assert len(prediction_jsons) > 0, f"No prediction files found in {self.prediction_path}"
        
        for prediction in tqdm(prediction_jsons, 'Processing prediction'):            
            # print(prediction)
            single_prediction = {}
            try:
                with open(prediction, 'rb') as f:
                    prediction_dict = json.load(f)
                
                if 'ious' not in prediction_dict:
                    print(f"Warning: No 'ious' field in {prediction}, skipping...")
                    continue
                    
                pred_iou = prediction_dict['ious'][0]
                
                assert isinstance(pred_iou, float), f"Expected pred_iou to be float, got {type(pred_iou)}"
                frame_id = prediction_dict['id']
                num_objects = self.get_meta_info(os.path.join(SEED_PATH, frame_id, 'meta_info.json'))
                platform = self.get_platform(prediction)
                single_prediction.update({
                    "iou": pred_iou,
                    "num_objects": num_objects,
                    "platform": platform,
                })
                results_list.append(single_prediction)
            except Exception as e:
                print(f"Error processing {prediction}: {e}")
                continue

        for single_result in results_list:
            object_num = single_result['num_objects']
            record = getattr(self, self.get_record_name(object_num))
            record.record_single(single_result)
            # Also record to the main metric recorder for the table display
            self.metric_recoder.record_single(single_result)

        # Display the object count table format
        print("\n" + "="*80)
        print("RESULTS BY OBJECT COUNT RANGES")
        print("="*80)
        self.metric_recoder.display_object_count_table()
        
        # final_rel_dict = {}
        # for i in self.object_num_split:
        #     record = getattr(self, f'metric_recoder_{str(i).zfill(2)}')
        #     miou_result, acc_result = record.display_mertic()
        #     final_rel_dict.update({
        #         f'{i}_metrics': {
        #             'iou_results': miou_result,
        #             'acc_results': acc_result
        #         }
        #     })
        # with open(f"{self.prediction_path}/metric.json", 'w', encoding='utf-8') as f:
        # # with open(f"exps/record_results/butd_{self.config.modality}_{self.split_checkpoint}.json", 'w', encoding='utf-8') as f:
        #     json.dump(final_rel_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    tester = Tester(prediction_path='code/3eed/butd_logs/Train_quad_drone_waymo_Val_quad_drone_waymo/predictions/')
    tester.calculate_metric()