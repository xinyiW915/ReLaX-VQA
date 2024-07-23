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

# pre-trained ResNet-50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

# to device
resnet50 = models.resnet50(pretrained=True).to(device)

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
    conv_idx = layer_name
    layer_obj = eval(conv_idx)

    activations = get_activation(resnet50, layer_obj, input_img_data)
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

    png_path = f'../visualisation/resnet50/{video_name}/frame_{frame_number}/'
    npy_path = f'../features/resnet50/{video_name}/frame_{frame_number}/'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(npy_path, exist_ok=True)

    if qp == "original_ugc":
        fig_name = f"resnet50_feature_map_original_layer_{layer_name}"
        combined_name = f"resnet50_feature_map_original"
    else:
        fig_name = f"resnet50_feature_map_qp_{qp}_layer_{layer_name}"
        combined_name = f"resnet50_feature_map_qp_{qp}"

    activated_img, activation_array = get_activation_map(image_path, layer_name, device)

    # save activation maps as png
    # get_activation_png(png_path, fig_name, activated_img)
    # save activation features as pny
    # get_activation_npy(npy_path, fig_name, activation_array)
    logger_setup.logger.debug(f'Shape of activation array: {activation_array.shape}')

    frame_npy_path = f'../features/resnet50/{video_name}/frame_{frame_number}_{combined_name}.npy'
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
    for idx, (name, layer) in enumerate(resnet50.named_children()):
        print(f"Index: {idx}, Layer Name: {name}, Layer Type: {type(layer)}")

    layer_name = 'layer4.2.conv2'
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



# # ResNet-50 layers to visualize
# layers_to_visualize_resnet50 = {
#     'conv1': 0,
#     'layer1.0.conv1': 2,
#     'layer1.0.conv2': 3,
#     'layer1.1.conv1': 5,
#     'layer1.1.conv2': 6,
#     'layer1.2.conv1': 8,
#     'layer1.2.conv2': 9,
#     'layer2.0.conv1': 11,
#     'layer2.0.conv2': 12,
#     'layer2.1.conv1': 14,
#     'layer2.1.conv2': 15,
#     'layer2.2.conv1': 17,
#     'layer2.2.conv2': 18,
#     'layer2.3.conv1': 20,
#     'layer2.3.conv2': 21,
#     'layer3.0.conv1': 23,
#     'layer3.0.conv2': 24,
#     'layer3.0.downsample.0': 25,
#     'layer3.1.conv1': 27,
#     'layer3.1.conv2': 28,
#     'layer3.2.conv1': 30,
#     'layer3.2.conv2': 31,
#     'layer3.3.conv1': 33,
#     'layer3.3.conv2': 34,
#     'layer4.0.conv1': 36,
#     'layer4.0.conv2': 37,
#     'layer4.0.downsample.0': 38,
#     'layer4.1.conv1': 40,
#     'layer4.1.conv2': 41,
#     'layer4.2.conv1': 43,
#     'layer4.2.conv2': 44,
# }

# Index: 0, Layer Name: conv1, Layer Type: <class 'torch.nn.modules.conv.Conv2d'>
# Index: 1, Layer Name: bn1, Layer Type: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
# Index: 2, Layer Name: relu, Layer Type: <class 'torch.nn.modules.activation.ReLU'>
# Index: 3, Layer Name: maxpool, Layer Type: <class 'torch.nn.modules.pooling.MaxPool2d'>
# Index: 4, Layer Name: layer1, Layer Type: <class 'torch.nn.modules.container.Sequential'>
# Index: 5, Layer Name: layer2, Layer Type: <class 'torch.nn.modules.container.Sequential'>
# Index: 6, Layer Name: layer3, Layer Type: <class 'torch.nn.modules.container.Sequential'>
# Index: 7, Layer Name: layer4, Layer Type: <class 'torch.nn.modules.container.Sequential'>
# Index: 8, Layer Name: avgpool, Layer Type: <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>
# Index: 9, Layer Name: fc, Layer Type: <class 'torch.nn.modules.linear.Linear'>
