# ReLaX-VQA
![visitors](https://visitor-badge.laobi.icu/badge?page_id=xinyiW915/ReLaX-VQA) 
[![GitHub stars](https://img.shields.io/github/stars/xinyiW915/ReLaX-VQA?style=social)](https://github.com/xinyiW915/ReLaX-VQA)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2407.11496-b31b1b.svg)](https://arxiv.org/abs/2407.11496)

Official Code for the following paper:

**X. Wang, A. Katsenou, and D. Bull**. [ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment](https://arxiv.org/abs/2407.11496)

The paper “[Frame Differences Matter in Quality Assessment of Compressed Videos](https://ieeexplore.ieee.org/document/11075040)” was accepted by the IEEE 25th International Conference on Digital Signal Processing ([DSP 2025](https://2025.ic-dsp.org/)).

Try our online demo on Hugging Face 🤗: [https://huggingface.co/spaces/xinyiW915/ReLaX-VQA](https://huggingface.co/spaces/xinyiW915/ReLaX-VQA)

---
[//]: # (## Abstract)
[//]: # (With the rapid growth of User-Generated Content &#40;UGC&#41; exchanged between users and sharing platforms, the need for video quality assessment in the wild has emerged. UGC is mostly acquired using consumer devices and undergoes multiple rounds of compression or transcoding before reaching the end user. Therefore, traditional quality metrics that require the original content as a reference cannot be used. In this paper, we propose ReLaX-VQA, a novel No-Reference Video Quality Assessment &#40;NR-VQA&#41; model that aims to address the challenges of evaluating the diversity of video content and the assessment of its quality without reference videos. ReLaX-VQA uses fragments of residual frames and optical flow, along with different expressions of spatial features of the sampled frames, to enhance motion and spatial perception. Furthermore, the model enhances abstraction by employing layer-stacking techniques in deep neural network features &#40;from Residual Networks and Vision Transformers&#41;. Extensive testing on four UGC datasets confirms that ReLaX-VQA outperforms existing NR-VQA methods with an average SRCC value of 0.8658 and PLCC value of 0.8872. We will open source the code and trained models to facilitate further research and applications of NR-VQA.)


## Performance
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relax-vqa-residual-fragment-and-layer-stack/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=relax-vqa-residual-fragment-and-layer-stack)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relax-vqa-residual-fragment-and-layer-stack/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=relax-vqa-residual-fragment-and-layer-stack)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relax-vqa-residual-fragment-and-layer-stack/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=relax-vqa-residual-fragment-and-layer-stack)

We evaluate the performance of ReLaX-VQA on four datasets. ReLaX-VQA has three different versions based on the training and testing strategies:
-  **ReLaX-VQA**: Trained and tested on each dataset with an **80%-20% random split**.
-  **ReLaX-VQA (w/o FT)**: Trained on **[LSVQ](https://github.com/baidut/PatchVQ)**, and the frozen model was tested on other datasets.
-  **ReLaX-VQA (w/ FT)**: Trained on **[LSVQ](https://github.com/baidut/PatchVQ)**, and the frozen model was **fine-tuned** on other datasets.

#### **Spearman’s Rank Correlation Coefficient (SRCC)**
| Model                 | CVD2014 | KoNViD-1k | LIVE-VQC | YouTube-UGC |
|-----------------------|--------|--------|--------|--------|
| ReLaX-VQA             | 0.8643 | 0.8535 | 0.7655 | 0.8014 |
| ReLaX-VQA (w/o FT)    | 0.7845 | 0.8312 | 0.7664 | 0.8104 |
| **ReLaX-VQA (w/ FT)** | **0.8974** | **0.8720** | **0.8468** | **0.8469** |

#### **Pearson’s Linear Correlation Coefficient (PLCC)**  
| Model                 | CVD2014    | KoNViD-1k | LIVE-VQC | YouTube-UGC |
|-----------------------|------------|----------|----------|-------------|
| ReLaX-VQA             | 0.8895     | 0.8473   | 0.8079   | 0.8204      |
| ReLaX-VQA (w/o FT)    | 0.8336     | 0.8427   | 0.8242   | 0.8354      |
| **ReLaX-VQA (w/ FT)** | **0.9294** | **0.8668** | **0.8876** | **0.8652**  |

More results can be found in **[reported_result.ipynb](https://github.com/xinyiW915/ReLaX-VQA/blob/main/reported_result.ipynb)**.


## Proposed Model
The figure shows the overview of the proposed ReLaX-VQA framework. The architectures of ResNet-50 Stack (I) and ResNet-50 Pool (II) are provided in Fig.2 in the paper.

<img src="./Framework.png" alt="proposed_ReLaX-VQA_framework" width="800"/>


## Usage
### 📌 Install Requirement
The repository is built with **Python 3.10.14** and can be installed via the following commands:

```shell
git clone https://github.com/xinyiW915/ReLaX-VQA.git
cd ReLaX-VQA
conda create -n relaxvqa python=3.10.14 -y
conda activate relaxvqa
pip install -r requirements.txt  
```


### 📥 Download UGC Datasets

The corresponding raw video datasets can be downloaded from the following sources:  
[LSVQ](https://github.com/baidut/PatchVQ), [KoNViD-1k](https://database.mmsp-kn.de/konvid-1k-database.html), [LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/), [YouTube-UGC](https://media.withyoutube.com/), [CVD2014](https://qualinet.github.io/databases/video/cvd2014_video_database/).  

The metadata for the experimented UGC dataset is available under [`./metadata`](./metadata).  

Once downloaded, place the datasets in [`./ugc_original_videos`](./ugc_original_videos) or any other storage location of your choice.  
Ensure that the `video_path` in the [`get_video_paths`](./src/main_relaxvqa_feats.py) function inside `main_relaxvqa_feats.py` is updated accordingly.


### 🎬 Test Demo  
Run the pre-trained models to evaluate the quality of a single video.  

The model weights provided in [`./model`](./model) contain the best-performing saved weights from training.  

To evaluate the quality of a specific video, run the following command:
```shell
python demo_test_gpu.py 
    -device <DEVICE> 
    -train_data_name <TRAIN_DATA_NAME> 
    -is_finetune <True/False> 
    -save_path <MODEL_PATH> 
    -video_type <DATASET_NAME> 
    -video_name <VIDEO_NAME> 
    -framerate <FRAMERATE>
```
Or simply try our demo video by running:
```shell
python demo_test_gpu.py
```

## Training  
Steps to train ReLaX-VQA from scratch on different datasets.  

### Extract Features  
Run the following command to extract features from videos:   
```shell
python main_relaxvqa_feats.py -device gpu -video_type youtube_ugc
```


### Train Model  
Train our model using extracted features:
```shell
python model_regression_simple.py -data_name youtube_ugc -feature_path ../features/ -save_path ../model/
```


For **LSVQ**, train the model using:  
```shell
python model_regression.py -data_name lsvq_train -feature_path ../features/ -save_path ../model/
```



### Fine-Tuning
To fine-tune the pre-trained model on a new dataset, modify [`train_data_name`](./src/model_finetune.py)
to match the dataset used for training, and [`test_data_name`](./src/model_finetune.py) to specify the dataset for fine-tuning.  
```shell
python model_finetune.py
```


## Ablation Study  
A detailed analysis of different components in ReLaX-VQA.

### Spatio-Temporal Fragmentation & DNN Layer Stacking  
Key techniques used in ReLaX-VQA:  
- **Fragmentation with DNN layer stacking:**  
  ```shell
  python feature_fragment_layerstack.py
  ```
- **Fragmentation with DNN layer pooling:**  
  ```shell
  python feature_fragment_pool.py
  ```

- **Frame with DNN layer stacking:**  
  ```shell
  python feature_layerstack.py
  ```

- **Frame with DNN layer pooling:**  
  ```shell
  python feature_pool.py
  ```

### Other Utilities  
#### Excluding Greyscale Videos
We exclude greyscale videos in our experiments. You can use [`check_greyscale.py`](./src/data_processing/check_greyscale.py)to filter out greyscale videos from the VQA dataset you want to use.
```shell
python check_greyscale.py
```


#### Metadata Extraction  
For easy extraction of metadata from your VQA dataset, use:
```shell
python extract_metadata_NR.py
```


## Acknowledgment
This work was funded by the UKRI MyWorld Strength in Places Programme (SIPF00006/1) as part of my PhD study.

## Citation
If you find this paper and the repo useful, please cite our paper 😊:

```bibtex
@article{wang2024relax,
      title={ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment},
      author={Wang, Xinyi and Katsenou, Angeliki and Bull, David},
      year={2024},
      eprint={2407.11496},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.11496}}

@INPROCEEDINGS{11075040,
  author={Wang, Xinyi and Katsenou, Angeliki and Bull, David},
  booktitle={2025 25th International Conference on Digital Signal Processing (DSP)}, 
  title={Frame Differences Matter in Quality Assessment of Compressed Videos}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={User-generated content;Digital signal processing;Transcoding;Video compression;Transformers;Quality assessment;Spatiotemporal phenomena;Video recording;Testing;Residual neural networks;No-Reference Video Quality Assessment;User-generated Content;Deep Features;Video Compression},
  doi={10.1109/DSP65409.2025.11075040}}
```

## Contact:
Xinyi WANG, ```xinyi.wang@bristol.ac.uk```
