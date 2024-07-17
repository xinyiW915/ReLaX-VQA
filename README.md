# ReLaX-VQA
will update it soon..

This repository will open source the code from the following paper: 
X. Wang, A. Katsenou, and D. Bull,  ["ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment"](https://arxiv.org/abs/2407.11496v1)

## Abstract
With the rapid growth of User-Generated Content (UGC) exchanged between users and sharing platforms, the need for video quality assessment in the wild has emerged. UGC is mostly acquired using consumer devices and undergoes multiple rounds of compression or transcoding before reaching the end user. Therefore, traditional quality metrics that require the original content as a reference cannot be used. In this paper, we propose ReLaX-VQA, a novel No-Reference Video Quality Assessment (NR-VQA) model that aims to address the challenges of evaluating the diversity of video content and the assessment of its quality without reference videos. ReLaX-VQA uses fragments of residual frames and optical flow, along with different expressions of spatial features of the sampled frames, to enhance motion and spatial perception. Furthermore, the model enhances abstraction by employing layer-stacking techniques in deep neural network features (from Residual Networks and Vision Transformers). Extensive testing on four UGC datasets confirms that ReLaX-VQA outperforms existing NR-VQA methods with an average SRCC value of 0.8658 and PLCC value of 0.8872. We will open source the code and trained models to facilitate further research and applications of NR-VQA: [this GitHub repository](https://github.com/xinyiW915/ReLaX-VQA).


## Methodology
<img src="./Framework.png" alt="proposed_ReLaX-VQA_framework" width="800"/>

The figure shows the overview of the proposed ReLaX-VQA framework. The architectures of ResNet-50 Stack (I) and ResNet-50 Pool (II) are provided in Fig.2 in the Appendix.


## Citation
If you use the code provided in this repository, please cite our paper:

```bibtex
@misc{wang2024relaxvqaresidualfragmentlayer,
      title={ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment}, 
      author={Xinyi Wang and Angeliki Katsenou and David Bull},
      year={2024},
      eprint={2407.11496},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.11496}, 
}


[![arXiv](https://img.shields.io/badge/arXiv-2407.11496-b31b1b.svg)](https://arxiv.org/abs/2407.11496)
## Acknowledgment
This work was funded by the UKRI MyWorld Strength in Places Programme (SIPF00006/1).
