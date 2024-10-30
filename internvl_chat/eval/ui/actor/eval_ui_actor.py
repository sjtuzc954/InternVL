import argparse
import json, jsonlines
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os, sys
import re
import pprint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_coordinate(coord):
    x = 0
    y = 0
    valid = True
    coord_pattern = re.compile(r"\[(\d+), (\d+)\]")
    match = coord_pattern.match(coord)
    if not match:
        valid = False
    else:
        x, y = match.groups()
        x, y = int(x), int(y)
        valid = 0 <= x <= 1000 and 0 <= y <= 1000
    return valid, x, y
    

def eval_response(annotation, responses):
    pattern = re.compile(r"操作：(.*)\n参数：(.*)")
    scroll_pattern = re.compile(r"(.*) -> (.*)")
    invalid_format = 0
    action_mismatch = 0
    invalid_param = 0
    input_mismatch = 0
    ae_x = []
    ae_y = []
    mae_x = 0
    mae_y = 0
    for (question, label), response in zip(annotation, responses):
        print(f"Evaluating response for {question}")
        # match = re.search(pattern, response)
        match = pattern.match(response)
        if not match:
            invalid_format += 1
            continue
        # else:
        #     print(match.groups())
        #     print(match.group(0))
        #     print(match.group(1))
        action, param = match.groups()
        # param = match.group(1)
        match = pattern.match(label)
        if not match:
            raise ValueError(f"Invalid label: {label}")
        action_truth, param_truth = match.groups()
        if action != action_truth:
            action_mismatch += 1
            print(f"Action mismatch. Expected: {action_truth}, Actual: {action}")
        elif action == "点击" or action == "长按":
            valid, x, y = parse_coordinate(param)
            if not valid:
                invalid_param += 1
                print(f"Invalid param: {param}")
                continue
            valid, x_truth, y_truth = parse_coordinate(param_truth)
            if not valid:
                raise ValueError(f"Invalid label: {label}")
            ae_x.append(abs(x - x_truth))
            ae_y.append(abs(y - y_truth))
        elif action == "滑动":
            match = scroll_pattern.match(param)
            if not match:
                invalid_param += 1
                print(f"Invalid param: {param}")
                continue
            valid1, x1, y1 = parse_coordinate(match.groups()[0])
            valid2, x2, y2 = parse_coordinate(match.groups()[1])
            if not valid1 or not valid2:
                invalid_param += 1
                print(f"Invalid param: {param}")
                continue
            match = scroll_pattern.match(param_truth)
            if not match:
                raise ValueError(f"Invalid label: {label}")
            valid1, x1_truth, y1_truth = parse_coordinate(match.groups()[0])
            valid2, x2_truth, y2_truth = parse_coordinate(match.groups()[1])
            if not valid1 or not valid2:
                raise ValueError(f"Invalid label: {label}")
            ae_x.append(abs(x1 - x1_truth))
            ae_x.append(abs(x2 - x2_truth))
            ae_y.append(abs(y1 - y1_truth))
            ae_y.append(abs(y2 - y2_truth))
        elif action == "输入":
            if param != param_truth:
                input_mismatch += 1
                print(f"Input mismatch. Expected: {param_truth}, Actual: {param}")
    if len(ae_x) != 0:
        mae_x = sum(ae_x) / len(ae_x)
    if len(ae_y) != 0:
        mae_y = sum(ae_y) / len(ae_y)
    return dict(
        invalid_format=invalid_format,
        action_mismatch=action_mismatch,
        invalid_param=invalid_param,
        input_mismatch=input_mismatch,
        mae=(mae_x, mae_y)
    )
    
def evaluate(ckpt, meta_path):
    with open(meta_path, encoding="UTF-8") as f:
        metadata = json.load(f)
        root = metadata['ui-dataset']['root']
        annotation_file = metadata['ui-dataset']['annotation']
    annotation = []
    responses = []
    model = AutoModel.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    with open(annotation_file, encoding="UTF-8") as f:
        for jsonl in jsonlines.Reader(f):
            if 'image' not in jsonl:
                raise Exception("field 'image' is required")
            image_path = jsonl['image']
            pixel_values = load_image(os.path.join(root, image_path), max_num=12).to(torch.bfloat16).cuda()
            question = jsonl['conversations'][0]['value']
            label = jsonl['conversations'][1]['value']
            response = model.chat(tokenizer, pixel_values, question, generation_config).strip()
            print(f"User: {question}")
            print(f"Assistant: {response}")
            annotation.append((question, label))
            responses.append(response)
    
    return eval_response(annotation, responses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--meta_path', type=str, required=True)
    
    args = parser.parse_args()
    
    # print(eval_response([("", "操作：点击\n参数：[487, 366]")], ["操作：点击\n参数：[481, 472]"]))
    sys.stdout = open("eval/ui/actor/log_e2e.txt", "w")
    result = evaluate(args.checkpoint, args.meta_path)
    with open("eval/ui/actor/results_e2e.txt", "w") as f:
        pprint.pprint(result, stream=f)
    
    