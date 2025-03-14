import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from thop import profile


# get the activation
def get_activation(model, layer, input_img_data):
    model.eval()
    activations = []
    inputs = []

    def hook(module, input, output):
        activations.append(output)
        inputs.append(input[0])

    hook_handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_img_data)
    hook_handle.remove()
    return activations, inputs

def get_activation_map(frame, layer_name, vgg16, device):
    # image pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations (resize and normalize)
    frame_tensor = transform(frame)

    # adding index 0 changes the original [C, H, W] shape to [1, C, H, W]
    if frame_tensor.dim() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
    # print(f'Image dimension: {frame_tensor.shape}')

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
    activations, inputs = get_activation(vgg16, layer_obj, frame_tensor)
    activated_img = activations[0][0]
    activation_array = activated_img.cpu().numpy()

    # calculate FLOPs for layer
    flops, params = profile(layer_obj, inputs=(inputs[0],), verbose=False)
    if params == 0 and isinstance(layer_obj, torch.nn.Conv2d):
        params = layer_obj.in_channels * layer_obj.out_channels * layer_obj.kernel_size[0] * layer_obj.kernel_size[1]
        if layer_obj.bias is not None:
            params += layer_obj.out_channels
    # print(f"FLOPs for {layer_name}: {flops}, Params: {params}")
    return activated_img, activation_array, flops, params

def process_video_frame(video_name, frame, frame_number, layer_name, vgg16, device):
    # create a dictionary to store activation arrays for each layer
    activations_dict = {}
    total_flops = 0
    total_params = 0
    fig_name = f"vgg16_feature_map_layer_{layer_name}"
    combined_name = f"vgg16_feature_map"

    activated_img, activation_array, flops, params = get_activation_map(frame, layer_name, vgg16, device)
    total_flops += flops
    total_params += params

    # save activation maps as png
    # png_path = f'../visualisation/vgg16/{video_name}/frame_{frame_number}/'
    # npy_path = f'../features/vgg16/{video_name}/frame_{frame_number}/'
    # os.makedirs(png_path, exist_ok=True)
    # os.makedirs(npy_path, exist_ok=True)
    # get_activation_png(png_path, fig_name, activated_img)
    # save activation features as pny
    # get_activation_npy(npy_path, fig_name, activation_array)

    # print(f"total FLOPs for Resnet50 layerstack: {total_flops}, Params: {total_params}")
    frame_npy_path = f'../features/vgg16/{video_name}/frame_{frame_number}_{combined_name}.npy'
    return activated_img, frame_npy_path, total_flops, total_params

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
    device_name = "gpu"
    if device_name == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    # pre-trained VGG16 model to device
    vgg16 = models.vgg16(pretrained=True).to(device)

    for idx, layer in enumerate(vgg16.features):
        print(f"Index: {idx}, Layer Type: {type(layer)}")
    layer_name = 'fc2'

    video_type = 'test'
    # Test
    if video_type == 'test':
        metadata_path = "../../metadata/test_videos.csv"
    # NR:
    elif video_type == 'resolution_ugc':
        resolution = '360P'
        metadata_path = f"../../metadata/YOUTUBE_UGC_{resolution}_metadata.csv"
    else:
        metadata_path = f'../../metadata/{video_type.upper()}_metadata.csv'

    ugcdata = pd.read_csv(metadata_path)
    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]
        sampled_frame_path = os.path.join('../..', 'video_sampled_frame', 'sampled_frame', f'{video_name}')

        print(f"Processing video: {video_name}")
        image_paths = glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
        frame_number = 0
        for image in image_paths:
            print(f"{image}")
            frame_number += 1
            process_video_frame(video_name, image, frame_number, layer_name, vgg16, device)

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