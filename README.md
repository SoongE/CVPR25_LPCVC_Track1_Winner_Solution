# Layer Imitation

This is the official winning solution of
**[2025 IEEE Low Power Computer Vision Challenge Track1](https://lpcv.ai/). (LPCV 2025)**
*LabLVM*

LPCV2025 is the 8th workshop on Efficient Deep Learning for Computer Vision at CVPR 2025.

The repository contains code for layer imitation training, evaluation, and submission to AIhub.

### Highlights

- We use MobileCLIP-S1 as the base model for layer imitation.
- Our approach reduces inference time by 13% and the number of parameters by 15%, with only a 1% decrease in accuracy on
  our custom dataset.
- We do not use any synthetic images; fine-tuning and testing are performed exclusively on the COCO 2014 and COCO 2017
  datasets.

## Getting Started

### Setup

Our experiments are conducted using PyTorch 2.3.0 and CUDA 12.1.
Install the required packages by running the following command.

```commandline
pip install -e .
```

To download the MobileCLIP-S1 weight from [ml-mobileclip](https://github.com/apple/ml-mobileclip/tree/main),
run the command below.

```commandline
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt -P weight
```

To build text embedding, run the command below.

```commandline
python extract_text_features.py
```

### Dataset

Download the COCO datasets and organize them according to the directory structure shown below.

```commandline
coco
  ├── train2014
  ├── val2014
  ├── test2014
  ├── train2017
  ├── val2017
  └── test2017
```

### Layer Imitation Train

To imitate the original layer, run `train.py`. The model weight will be saved to `output` directory.

```commandline
python train.py 
```

### Evaluation

To reproduce the results, run `eval.py`.

```commandline
python eval.py 
```

The results are evaluated on a custom dataset composed of COCO 2014 and COCO 2017.

|          | Parameters (M) | Throughput (ms) | Accuracy (%) |
|----------|----------------|-----------------|--------------|
| Baseline | 20.5           | 2.3             | 47.5         |
| Ours     | 17.4           | 2.0             | 47.0         |

### Submit to AIhub

Run `run.py` to submit to AIhub.

```commandline
python run.py 
```

## Acknowledgements
- [ml-mobileclip](https://github.com/apple/ml-mobileclip/tree/main?tab=readme-ov-file)
