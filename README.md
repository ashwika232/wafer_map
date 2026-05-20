# Image-Based Wafer Map Pattern Intelligence  
### Deep Learning for Semiconductor Defect Classification

---

## Overview  
This project presents an AI-driven approach for analyzing semiconductor wafer maps using deep learning. The system automatically detects and classifies defect patterns such as Edge-Ring, Center, Scratch, Donut, and Random defects, enabling faster fault diagnosis and improved yield prediction.

Traditional inspection methods struggle with complex spatial patterns. This project leverages computer vision and deep learning techniques to improve accuracy and automation.

---

## Key Features  
- Automated wafer defect classification  
- High-performance CNN and transfer learning models  
- Handles imbalanced real-world datasets  
- Visual explanations using Grad-CAM  
- Scalable for industrial applications  

---

## System Architecture  

Wafer Map Image
        ↓
Preprocessing (Resize, Normalize, Augment)
        ↓
Deep Learning Model (CNN / ResNet)
        ↓
Feature Extraction
        ↓
Classification Layer
        ↓
Output (Defect Type + Confidence Score)


---

## Methodology  

### Data Collection  
- Dataset: WM-811K Wafer Map Dataset  
- Contains labeled defect patterns  

### Data Preprocessing  
- Image resizing (64×64 or 128×128)  
- Normalization  
- Data augmentation (rotation, flipping, noise)  
- Handling class imbalance  

### Model Development  
- Custom CNN architecture  
- Transfer learning (ResNet / MobileNet)  
- Training using cross-entropy loss  



---

## Model Architecture (Example CNN)

```python
class WaferCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,9)
        )
```


##Results

Accurate classification of wafer defect patterns

Effective detection of spatial defect structures

Robust performance on imbalanced datasets

<img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/2bc99ba7-c045-46a5-9840-b5b43806366d" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/3125c204-156a-4d4f-a6cc-19c5344a4839" />
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 64, 64, 16)     │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 64, 64, 16)     │            64 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (ReLU)                    │ (None, 64, 64, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 32, 32, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 32, 32, 32)     │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 32, 32, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (ReLU)                  │ (None, 32, 32, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 16, 16, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 16, 16, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 16, 16, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_2 (ReLU)                  │ (None, 16, 16, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 8, 8, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 4096)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ feature_layer (Dense)           │ (None, 256)            │     1,048,832 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_3 (ReLU)                  │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 8)              │         2,056 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,223,450 (12.30 MB)
 Trainable params: 1,074,408 (4.10 MB)
 Non-trainable params: 224 (896.00 B)
 Optimizer params: 2,148,818 (8.20 MB)



