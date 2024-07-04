"""
copied from
https://github.com/LiheYoung/Depth-Anything/issues/36
"""

# infer.py
# Code by @1ssb
import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Global settings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 256  # image height
FINAL_WIDTH = 256  # image shape
P_x, P_y = 128, 128  # center of the image
INPUT_DIR = './my_test/input'
OUTPUT_DIR = './my_test/output'
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader


def to_point_cloud(resized_pred, focal_length_x, focal_length_y, FINAL_WIDTH, FINAL_HEIGHT):
    """
    resized_pred は(FINAL_WIDTH, FINAL_HEIGHT)の大きさに既になっている。
    そのため、resized_predの中には既に画像サイズが入っている。
    """
    P_x = FINAL_WIDTH // 2 # center of the image
    P_y = FINAL_HEIGHT // 2
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - P_x) / focal_length_x
    y = (y - P_y) / focal_length_y
    z = np.array(resized_pred)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    return points

def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()

            # Resize color image and depth to final size
            resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
            resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

            focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
            points = to_point_cloud(resized_pred, focal_length_x, focal_length_y, FINAL_WIDTH, FINAL_HEIGHT)
            colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
