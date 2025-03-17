import os
import time
import sys
import argparse
from multiprocessing import Pool
from transformers import pipeline
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from tqdm import tqdm

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objaverse_df = pd.read_csv('../qa_results/objaverse_df_with_labels_and_scores.csv')

    # start = args.objects_start
    # end = start + args.num_objects
    # objaverse_df = objaverse_df.iloc[start:end]
    # Fill missing captions 
    objaverse_df['trellis_aesthetic_score'] = objaverse_df['trellis_aesthetic_score'].fillna(0)
    objaverse_df['trellis_caption'] = objaverse_df['trellis_caption'].fillna('')
    objaverse_df['cap3D_caption'] = objaverse_df['cap3D_caption'].fillna('')


    classifier_pipe = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli", device="cuda")

    category_dict = {
        1: 'single high quality 3D object', 2: 'a scene or several objects'
    }
    is_car_dict = {
        1: 'car', 2: 'not a car'
    }
    is_vehicle_dict_2 = {
        1: 'vehicle', 2: 'not a vehicle'
    }
    # 1: 'vehicle', 2: 'car', 3: 'not a vehicle', 4: 'not a car'
    candidate_labels_car = list(is_car_dict.values())
    candidate_labels_vehicle = list(is_vehicle_dict_2.values())
    candidate_labels_objects = list(category_dict.values())

    classified_caption_uids_df = pd.DataFrame()
    for uid, trellis_caption, cap3D_caption in tqdm(zip(objaverse_df['uid'], objaverse_df['trellis_caption'], objaverse_df['cap3D_caption']), total=len(objaverse_df)):
        if len(cap3D_caption) == 0 and len(trellis_caption) == 0:
            classified_caption_uids_df = pd.concat([classified_caption_uids_df, pd.DataFrame(
                [{'uid': uid, 'caption': '', 'descibes_car_score': -1.0, 'text_category_car': 'nan', 'descibes_vehicle_score': -1.0, 
                  'text_category_vehicle': 'nan', 'descibes_object_score': -1.0, 'text_category_object': 'nan'
                  }])], ignore_index=True)
            continue
        elif len(trellis_caption) > 0 and len(cap3D_caption) > 0:
            # combine both captions
            caption = trellis_caption[:-1] + ", \"" + cap3D_caption + "\"]"
        else:
            caption = trellis_caption if len(trellis_caption) > 0 else cap3D_caption

        car_score, car_text_category = classify_caption(caption, classifier_pipe, candidate_labels_car, 'car')
        vehicle_score, vehicle_text_category = classify_caption(caption, classifier_pipe, candidate_labels_vehicle, 'vehicle')
        is_object_score, object_text_category = classify_caption(caption, classifier_pipe, candidate_labels_objects, 'single high quality 3D object')
        classified_caption_uids_df = pd.concat([classified_caption_uids_df, pd.DataFrame(
            [{'uid': uid, 'caption': caption, 'descibes_car_score': car_score, 'text_category_car': car_text_category, 
              'descibes_vehicle_score': vehicle_score, 'text_category_vehicle': vehicle_text_category, 
              'descibes_object_score': is_object_score, 'text_category_object': object_text_category}])], ignore_index=True)
    
    classified_caption_uids_df.to_csv('./classified_caption_uids_df.csv', index=False)


def classify_caption(caption, classifier_pipe, candidate_labels, score_object):
    result = classifier_pipe(caption, candidate_labels)
    text_category = result['labels'][0]
    if result['labels'][0] == score_object:
        return_score = result['scores'][0]
    else:
        return_score = result['scores'][1]
    return return_score, text_category

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects_start", default=0, type=int)
    parser.add_argument("--num_objects", default=500000, type=int)
    args = parser.parse_args()

    main(args)

    