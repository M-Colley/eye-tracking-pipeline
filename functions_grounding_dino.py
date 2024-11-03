import os, sys

sys.path.append(os.path.join(os.getcwd(), ".\Grounded-Segment-Anything\GroundingDINO"))

import argparse
import copy

from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict , get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict #,load_image
from torchvision import transforms

import supervision as sv

# segment anything 2.1
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "sam2.1_hiera_large.pt"
model_cfg = "sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor_21 = SAM2ImagePredictor(sam2_model)

import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO

from huggingface_hub import hf_hub_download

from equilib import Equi2Pers

from colorsys import rgb_to_hsv

#from loguru import logger


# Define the transformation
transform = transforms.ToTensor()
# release all previous cuda stuff
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device used: " + str(device))
#device = "cpu"


# add relevant config-names
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
sam_checkpoint = "sam_vit_h_4b8939.pth"
#sam_model = build_sam(checkpoint=sam_checkpoint).to(device)
# old version
#sam_predictor = SamPredictor(sam_model)

plt.figure(figsize=(10, 10))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model
    

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)


# slightly different than in groundingdino utils; from HuggingFace
def load_image(image_path):
    # # load image
    if isinstance(image_path, PIL.Image.Image):
        image_pil = image_path
    else:
        image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)


        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)


    return image_pil, mask


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

def get_class_for_hex(hex_code, threshold=35):
    color_classes = {
        "#0f8071": "birdFlock",
        "#424d44": "explosion",
        "#0c1446": "newspaper",
        "#e6a6e7": "road",
        "#ec0e96": "display",
        "#e51010": "helicopter_interior",
        "#5635e5": "path",
        "#f5df91": "sky",
        "#22730f": "surroundings",
        "#34c43e": "person",
        # Add more mappings as needed
    }

    target_rgb = hex_to_rgb(hex_code)

    closest_class = "UnknownClass"
    min_distance = float("inf")

    for class_hex, class_name in color_classes.items():
        class_rgb = hex_to_rgb(class_hex)
        distance = color_distance(target_rgb, class_rgb)
        if distance < min_distance and distance < threshold:
            closest_class = class_name
            min_distance = distance

    return closest_class


def get_class_for_hex_exact(hex_code):
    print("hex_code: " + str(hex_code))
    color_classes = {
        "#0f8071": "birdFlock",
        "#424d44": "explosion",
        "#0c1446": "newspaper",
        "#e6a6e7": "road",
        "#ec0e96": "display",
        "#e51010": "helicopter_interior",
        "#5635e5": "path",
        "#f5df91": "sky",
        "#22730f": "surroundings",
        "#34c43e": "person",
        # Add more mappings as needed
    }
    return color_classes.get(hex_code, "UnknownClass")


def get_hex_code_at_position(pil_image, position):

    # Assuming position is a tuple (x, y)
    x, y = position
    width, height = pil_image.size
    r, g, b = 0, 0, 0
    
    if 0 <= x < width and 0 <= y < height:
        r, g, b = pil_image.getpixel(position)
    else:
        print("Position is out of bounds in get_hex_code_at_position!")
        print("position: " + str(position))
        print("pil_image.size: " + str(pil_image.size))
        print("Setting RGB to 0,0,0")
        
    
    # Convert to hex code
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return hex_code
    
    # Convert to hex code
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return hex_code

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)



# example: color = get_color_for_class("birds")
# Attention: has to be adapted per use case
def get_color_for_class(class_name):
    # from https://sashamaps.net/docs/resources/20-colors/
    # for trust-calibration: pedestrian . vehicle .  buildings . sky . display . road . symbol . roadside
    colors = {
            "car interior": np.array([128/255, 0/255, 0/255, 0.6]), # Maroon
            "pedestrian": np.array([128/255, 128/255, 0/255, 0.6]), # Olive
            "helicopters": np.array([0/255, 128/255, 128/255, 0.6]), # Teal
            "buildings": np.array([0/255, 0/255, 128/255, 0.6]), # Navy
            "vehicle": np.array([245/255, 130/255, 48/255, 0.6]), # Orange
            "sky": np.array([128/255, 128/255, 128/255, 0.6]), # Grey
            "road": np.array([70/255, 240/255, 240/255, 0.6]), # Cyan
            "roadside": np.array([70/255, 240/255, 240/255, 0.6]), # Cyan
            "newspaper": np.array([145/255, 30/255, 180/255, 0.6]), # Purple
            "symbol": np.array([255/255, 215/255, 180/255, 0.6]), # Apricot
            "display": np.array([60/255, 180/255, 75/255, 0.6]), # Green
    }

    # If the class name is found in the colors dictionary, return that color
    if class_name in colors:
        return colors[class_name]

    # Otherwise, return a default color for other generic terms
    return np.array([220/255, 190/255, 255/255, 0.6]) # Lavendel


def scale_to_fullHD(original_x, original_y, original_max_x, original_max_y):
    # Check if either value is negative
    if original_x < 0 or original_y < 0:
        return None, None
    
    # out of bounds
    if original_x > original_max_x or original_y > original_max_y:
        return None, None

    # we want to scale to FullHD
    target_max_x = 1920
    target_max_y = 1080

    scaled_x = (original_x / original_max_x) * target_max_x
    scaled_y = (original_y / original_max_y) * target_max_y


    return scaled_x, scaled_y

# copied from HuggingFace
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


  



def show_mask(mask, ax, phrase, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = get_color_for_class(phrase)
        #color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
    
    
def process_image(image_source, current_eye_gaze, text_prompt, return_segmented_image=False):
    # clear plot
    plt.clf()
    image_pil, image = load_image(image_source.convert("RGB"))
    size = image_pil.size

    # model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"

    # standards are box_threshold=0.30, text_threshold=0.25 but we want higher quality
    boxes_filt, pred_phrases = get_grounding_output(
            groundingdino_model, image = image, caption= text_prompt, box_threshold=0.35, text_threshold=0.30, device=device)
    if boxes_filt.size(0) == 0:
        print("No objects detected, please try others.")
        return "NULL", None
    
    print("pred_phrases: " + str(pred_phrases))

    #boxes_filt_ori = copy.deepcopy(boxes_filt)

    pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

    #image_with_box = plot_boxes_to_image(copy.deepcopy(image_pil), pred_dict)[0]

    image = np.array(image_source)
    predictor_21.set_image(image)

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.to(device)
    transformed_boxes = predictor_21.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    masks, _, _ = predictor_21.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
    # masks: [9, 1, 512, 512]
    assert sam_hq_checkpoint, 'sam_hq_checkpoint is not found!'
    
    skip_enumerate = False
        # the segmentation did not return anything
    if(masks == None):
        skip_enumerate = True

    # eye gaze was negative
    if current_eye_gaze is None or any(value is  None for value in current_eye_gaze):
        skip_enumerate = True

    # standard values
    detected_class = "NULL"

    if not skip_enumerate:
        if return_segmented_image:
            
            plt.imshow(image_source)
            for idx, mask in enumerate(masks):
                show_mask(mask.cpu().numpy(), plt.gca(), phrase = str(pred_phrases[idx])[:-6], random_color=False)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.cpu().numpy(), plt.gca(), label)

        for idx, mask in enumerate(masks): 
            #print("Getting color for: " + str(pred_phrases[idx])[:-6])
            # Give pixels of mask a color
            color = get_color_for_class(str(pred_phrases[idx])[:-6])
            h, w = mask.cpu().numpy().shape[-2:]
            mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
            print("looking at eye coordinate " + str(tuple(current_eye_gaze.T)))
            values_at_coords = mask_image[tuple(current_eye_gaze)]
            #print("values_at_coords: " + str(values_at_coords))
            if any(value != 0 for value in values_at_coords):
                #print("One value is not 0.")
                print("The current class is: " + str(pred_phrases[idx])[:-6])
                # You can use idx within this loop as the index of the current iteration
                detected_class = str(pred_phrases[idx])[:-6] # this assumes that the phrases and the masks are aligned
                break
            #else:
                #print("At least one value is 0.")

    if return_segmented_image:
        return detected_class, plt
    else:
        return detected_class, None


def calculate_view(frame, yaw, pitch):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    equi_img = np.asarray(frame)
    equi_img = np.transpose(equi_img, (2, 0, 1))

    # rotations
    rots = {
        'roll': 0.,
        'pitch': np.deg2rad(pitch),
        'yaw': np.deg2rad(yaw),
    }

    # Intialize equi2pers
    equi2pers = Equi2Pers(
        height=1080,
        width=1920,
        fov_x=110,
        mode="bilinear",
    )

    # obtain the perspective image
    pers_img = equi2pers(
        equi=equi_img,
        rots=rots,
    )

    # Transpose the image back to (height, width, channels)
    pers_img = np.transpose(pers_img, (1, 2, 0))

    # Convert to PIL Image
    pers_img_pil = Image.fromarray(pers_img.astype(np.uint8)) 
    
    return pers_img_pil


def process_frame_pre_segmented_360(outputDirs, frame_counter, frame, yaw, pitch, current_eye_gaze, save_image):
    pers_img_pil = calculate_view(frame, yaw, pitch)

    #print(pers_img_pil.size)
    #print(current_eye_gaze)
    # default value
    class_name = "NULL"

    if save_image == True:
        # Add a red-filled circle at a specific (x, y) position
        circle_radius = 8
        plt.imshow(pers_img_pil)
        plt.axis('off') # To turn off axes
        if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
            circle_x = current_eye_gaze[0]  
            circle_y = current_eye_gaze[1]  
            
            plt.scatter(circle_x, circle_y, color='red', s=circle_radius**2, alpha=0.7, edgecolors='none')
            plt.scatter(circle_x, circle_y, color='black', s=circle_radius**2, alpha=0.7, edgecolors='none', marker="+")

        output = os.path.join("EyeGazeOutput",outputDirs[0], outputDirs[1], outputDirs[2])
        if not os.path.exists(output):
            os.makedirs(output)
        plt.savefig(os.path.join(output, "testBild_" + str(frame_counter) +".png"), bbox_inches='tight', pad_inches=0)
        # pers_img_pil.save(os.path.join(output, "testBild_" + str(frame_counter) +".png"))
        plt.clf()
        print("image saved")

    if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
        hex_code = get_hex_code_at_position(pers_img_pil, current_eye_gaze)
        # You can then pass the hex_code to the get_class_for_hex function
        class_name = get_class_for_hex(hex_code)

    return class_name




def process_frame_360(frame, yaw, pitch, current_eye_gaze, text_prompt_custom, return_segmented_image):
    pers_img_pil = calculate_view(frame, yaw, pitch)

    if (return_segmented_image):
        detected_class, segmented_image = process_image(image_source=pers_img_pil, current_eye_gaze=current_eye_gaze, text_prompt=text_prompt_custom, model_t=groundingdino_model, return_segmented_image=True, sam_predictor=predictor_21)
        return detected_class, segmented_image
    else: 
        detected_class, _ = process_image(image_source=pers_img_pil, current_eye_gaze=current_eye_gaze, text_prompt=text_prompt_custom, model_t=groundingdino_model, return_segmented_image=False, sam_predictor=predictor_21)
        return detected_class, _
