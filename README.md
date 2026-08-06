

## Ultrasound-based Breast Nodule Malignancy Diagnosis using CNN enhanced by Knowledge Distillation

A deep learning framework integrating Dual-IB, hierarchical feature fusion, and knowledge distillation enables accurate, generalizable, and interpretable classification of benign and malignant breast ultrasound lesions. The final model includes:

- Student network: a pruned CNN backbone.
- Dual Inverted Bottleneck Block (Dual-IB).
- Hierarchical Feature Fusion (HF Fusion).
- Teacher model: DenseNet121.
- Loss function: LMF hybrid loss plus KL-based knowledge distillation loss, L_final=alpha.LMFLoss+(1-alpha).L_KD,


## Environment Setup

Recommended environment:

```bash
conda create -n ubnd python=3.10
conda activate ubnd
pip install -r requirements.txt
```

Main dependencies:

- Python >= 3.10
- PyTorch >= 2.0
- Torchvision >= 0.15
- OpenCV
- NumPy
- scikit-learn
- TensorBoard
- grad-cam

## Project Structure

```text
UBND/
|-- configs/                 
|-- datasets/                
|-- losses/                  
|-- metrics/                 
|-- models/
|   |-- backbone/           
|   |-- heads/              
|   |-- modules/             
|   |-- teacher/             
|   `-- network.py           
|-- trainers/               
|-- utils/                    
|-- visualization/                                                         
|-- requirements.txt
`-- README.md
```

## Data Preparation

The project supports two data organization formats.

### 1. ImageFolder Format

```text
data/BUSI/
|-- benign/
|   |-- xxx.png
|   `-- ...
`-- malignant/
    |-- yyy.png
    `-- ...
```

The default class mapping is:

```text
benign -> 0
malignant -> 1
```

### 2. TXT Annotation Format

Each line contains one image path and one class label:

```text
/path/to/image1.png 0
/path/to/image2.png 1
```


## Parameter Settings


- Input size: 224 x 224
- Batch size: 32
- Optimizer: AdamW
- Learning rate: 0.0008
- Weight decay: 1e-4
- Scheduler: Cosine Annealing
- Minimum learning rate: 1e-6
- Knowledge distillation temperature: T = 6
- Knowledge distillation weight: alpha = 0.5
- Teacher model: DenseNet121
- Focal Loss gamma: 2.0
- LDAM max_m: 0.5
- LDAM scale: 30.0
- Internal Focal/LDAM weight in LMF: 0.5

These supplementary settings can be modified in `configs/config.yaml`.


## Visualization

The manuscript uses the following CAM methods for interpretability analysis:

- GradCAM
- GradCAM++
- XGradCAM
- EigenGradCAM
- LayerCAM

This project provides a unified CAM interface in `visualization/cam.py`, based on the `grad-cam` package.

## Citation

```bibtex
@article{hu2026ubnd,
  title   = {Ultrasound-based Breast Nodule Malignancy Diagnosis using CNN enhanced by Knowledge Distillation},
  author  = {Anonymous Authors},
  journal = {iScience},
  year    = {2026}
}
```
