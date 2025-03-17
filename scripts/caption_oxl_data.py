import os
import requests
import argparse
import pandas as pd
import torch
from PIL import Image
import PIL.Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from quality_classifier.utils import get_images_from_zip

def main(args):
    render_location_df = pd.read_csv('../qa_results/combined_oxl_renders_df.csv') 
    objaverse_df = pd.read_csv('../qa_results/objaverse_df_with_labels_and_scores.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True)

    def run_example(pil_image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        image = pil_image.convert("RGB")
        inputs = processor(text=prompt, images=image,
                        return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer[prompt]

    def run_batched_example(pil_images, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        images = [image.convert("RGB") for image in pil_images]
        prompts = [prompt] * len(images)
        inputs = processor(text=prompts, images=images,
                        return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False)
        parsed_answerts = []
        for i, answer in enumerate(generated_text):
            parsed_answer = processor.post_process_generation(
                answer,
                task=task_prompt,
                image_size=(images[i].width, images[i].height)
            )
            parsed_answerts.append(parsed_answer[prompt])
        return parsed_answerts

    objaverse_df['trellis_caption'] = objaverse_df['trellis_caption'].fillna('')
    objaverse_df['cap3D_caption'] = objaverse_df['cap3D_caption'].fillna('')
    # get all uids where trellis_caption and cap3D_caption are empty
    uids = objaverse_df[(objaverse_df['trellis_caption'] == '') & (
        objaverse_df['cap3D_caption'] == '')]['uid']
    print(len(uids))
    # filter render_location_df for uids
    render_location_df_to_caption = render_location_df[render_location_df['object_id'].isin(uids)]
    
    start = args.objects_start
    end = start + args.num_objects
    if end > len(render_location_df_to_caption):
        end = len(render_location_df_to_caption)
    render_location_df_to_caption = render_location_df_to_caption.iloc[start:end]
    print(f"Captioning {len(render_location_df_to_caption)} objects")

    prompt_describe = "<DETAILED_CAPTION>"
    prompt_detect = '<OD>'

    all_captions_df = pd.DataFrame( columns=['uid', 'caption', 'OD_caption'])
    for df_item in tqdm(render_location_df_to_caption.iterrows(), total=len(render_location_df_to_caption)):
        img_path = df_item[1]['img_path']
        pil_images = get_images_from_zip(img_path)
        # captions = []
        # for pil_image in pil_images:
            # caption = run_example(pil_image, prompt)
            # captions.append(caption)
        captions = run_batched_example(pil_images, prompt_describe)
        captions = [caption.replace("<pad>", "") for caption in captions]
        caption_string = '; '.join(captions)
        od_captions = run_batched_example(pil_images, prompt_detect)
        od_caption_labels = [', '.join(caption['labels'])  for caption in od_captions]
        od_caption_label_string = ', '.join(od_caption_labels)
        uid = df_item[1]['object_id']
        captions_df = pd.DataFrame({'uid': uid, 'caption': caption_string, 'OD_caption': od_caption_label_string}, index=[0])
        all_captions_df = pd.concat([all_captions_df, captions_df], ignore_index=True)

    all_captions_df.to_csv(f'./captions_florence_{start}_{end}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects_start", default=0, type=int)
    parser.add_argument("--num_objects", default=122250, type=int)
    args = parser.parse_args()

    main(args)

    