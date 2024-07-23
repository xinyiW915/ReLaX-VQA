import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image

import utils.logger_setup as logger_setup

# pre-trained VGG16 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

# to device
vgg16 = models.vgg16(pretrained=True).to(device)

# get the activation
def get_activation(model, layer, input_img_data):
    model.eval()
    activations = []

    def hook(module, input, output):
        activations.append(output)

    hook_handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_img_data)

    hook_handle.remove()
    return activations

def get_activation_map(img_path, layer_name, device):
    # image pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path)
    img = transform(img)

    # adding index 0 changes the original [C, H, W] shape to [1, C, H, W]
    input_img_data = img.unsqueeze(0).to(device)
    # print(f'Image dimension: {input_img_data.shape}')

    # getting the activation of a given layer
    if layer_name == 'fc2' or layer_name == 'fc1':
        fc_idx = layer_name.replace('fc', '')
        if fc_idx == '2':
            fc_idx = int(fc_idx) + 1
        else:
            fc_idx = int(fc_idx) - 1
        layer_obj = vgg16.classifier[fc_idx]
    else:
        conv_idx = layer_name
        layer_obj = vgg16.features[conv_idx]

    activations = get_activation(vgg16, layer_obj, input_img_data)
    activated_img = activations[0][0]
    activation_array = activated_img.cpu().numpy()

    return activated_img, activation_array

def process_video_frame(video_name, image_path, layer_name, qp):
    filename = os.path.basename(image_path)
    if 'residual' in filename or 'next' in filename or 'ori' in filename:
        parts = filename.split('_')
        if 'residual_of' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
            if 'residual_of_imp' in filename:
                frame_number = parts[-4] + "_" + parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif 'residual_imp' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif '_residual_merged_frag' in filename:
            frame_number = parts[-4] + "_" + parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif 'ori_frag' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        else:
            frame_number = parts[-2] + "_" + parts[-1].split('.')[0]
    else:
        frame_number = int(filename.split('_')[-1].split('.')[0])

    png_path = f'../visualisation/vgg16/{video_name}/frame_{frame_number}/'
    npy_path = f'../features/vgg16/{video_name}/frame_{frame_number}/'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(npy_path, exist_ok=True)

    if qp == "original_ugc":
        fig_name = f"vgg16_feature_map_original_layer_{layer_name}"
        combined_name = f"vgg16_feature_map_original"
    else:
        fig_name = f"vgg16_feature_map_qp_{qp}_layer_{layer_name}"
        combined_name = f"vgg16_feature_map_qp_{qp}"

    activated_img, activation_array = get_activation_map(image_path, layer_name, device)

    # save activation maps as png
    # get_activation_png(png_path, fig_name, activated_img)
    # save activation features as pny
    # get_activation_npy(npy_path, fig_name, activation_array)
    logger_setup.logger.debug(f'Shape of activation array: {activation_array.shape}')

    frame_npy_path = f'../features/vgg16/{video_name}/frame_{frame_number}_{combined_name}.npy'
    return activation_array, frame_npy_path

def get_activation_png(png_path, fig_name, activated_img, n=8):
    fig = plt.figure(figsize=(10, 10))

    # visualise activation map for 64 channels
    for i in range(n):
        for j in range(n):
            idx = (n * i) + j
            if idx >= activated_img.shape[0]:
                break
            ax = fig.add_subplot(n, n, idx + 1)
            ax.imshow(activated_img[idx].cpu().numpy(), cmap='viridis')
            ax.axis('off')

    # save figures
    fig_path = f'{png_path}{fig_name}.png'
    print(fig_path)
    print("----------------" + '\n')
    plt.savefig(fig_path)
    plt.close()

def get_activation_npy(npy_path, fig_name, activation_array):
    np.save(f'{npy_path}{fig_name}.npy', activation_array)

if __name__ == '__main__':
    for idx, layer in enumerate(vgg16.features):
        print(f"Index: {idx}, Layer Type: {type(layer)}")

    layer_name = 'fc2'
    image_type = 'encoded_ugc' #original_ugc, encoded_ugc

    # for original video:
    if image_type == 'original_ugc':
        metadata_path = "../../metadata/YOUTUBE_UGC_metadata_original.csv"
        ugcdata = pd.read_csv(metadata_path)

    # for encoded video:
    elif image_type == 'encoded_ugc':
        codec_name = 'x264'
        metadata_path = f"../metadata/YOUTUBE_UGC_metadata_{codec_name}_metrics.csv"
        ugcdata = pd.read_csv(metadata_path)

    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]

        if image_type == 'original_ugc':
            qp = 'original'
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', 'original_sampled_frame', f'{video_name}')

        elif image_type == 'encoded_ugc':
            qp = ugcdata['QP'][i]
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', f'encoded_sampled_frame_qp_{qp}', f'{video_name}')

        print(f"Processing video: {video_name} at QP {qp}")
        image_paths = glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
        for image_path in image_paths:
            print(f"{image_path}")
            process_video_frame(video_name, image_path, layer_name, qp)

# layers_to_visualize = {
#     'conv1_1': 0,
#     'conv1_2': 2,
#     'conv2_1': 5,
#     'conv2_2': 7,
#     'conv3_1': 10,
#     'conv3_2': 12,
#     'conv3_3': 14,
#     'conv4_1': 17,
#     'conv4_2': 19,
#     'conv4_3': 21,
#     'conv5_1': 24,
#     'conv5_2': 26,
#     'conv5_3': 28,
# }

# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
#   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace=True)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True)
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace=True)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace=True)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)
#   (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )