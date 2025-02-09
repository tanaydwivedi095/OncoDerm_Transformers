# OncoDerm: Transformer-Based Phase

This phase of the OncoDerm project explores the use of **Transformer-based architectures** for skin cancer classification. Unlike traditional CNNs, Vision Transformers (ViTs) capture **global dependencies** in images, potentially improving classification accuracy.

## Overview
- Implements **Vision Transformers (ViTs)** for feature extraction and classification.
- Compares performance against CNN-based models from the previous phase.
- Fine-tunes transformer models on a skin cancer dataset to enhance classification accuracy.

## Dataset Details
The dataset consists of dermoscopic images labeled as either **Malignant** or **Benign**. Images undergo preprocessing, including:
- **Resizing** to match transformer input dimensions.
- **Normalization** to standardize pixel values.
- **Data Augmentation** to improve model generalization.

## Model Implemented
- **ViT-B16 (Vision Transformer Base-16)** fine-tuned with the **Adagrad optimizer**.
- The **B16 architecture** was replicated with some modifications to enhance performance.

### Why Vision Transformers?
Traditional CNNs rely on **local receptive fields**, limiting their ability to understand **long-range dependencies** within images. Vision Transformers process entire images using **self-attention mechanisms**, allowing them to:
- Capture **global spatial relationships**.
- Improve classification performance on complex patterns.
- Reduce reliance on extensive labeled datasets via **transfer learning**.

## Performance Evaluation
This model was able to achieve an **accuracy of 95.63%**.
Each model is evaluated using key metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-score**
- **ROC-AUC**

Results are compared with CNN-based models from the previous phase to assess improvements.

## Computational Requirements
- This model was designed for systems with **high computational power** but delivers **better classification results** compared to CNN-based approaches.

## Project Phases
1. **Transfer Learning (CNNs)**: Feature extraction using CNNs, combined with an **RFC classifier**. [🔗 Link](https://github.com/tanaydwivedi095/OncoDerm_Transfer_Learning)
2. **Transformer-Based Models (Current Phase)**: Exploring the power of self-attention mechanisms.
3. **Streamlit UI (Next Phase)**: Deploying a user-friendly interface for real-time classification. [🔗 Link](https://github.com/tanaydwivedi095/OncoDerm)

## Installation & Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/tanaydwivedi095/OncoDerm_Transformers.git
cd OncoDerm_Transformers
```

### Step 2: Install Dependencies
Since `requirements.txt` is not available, install the required packages manually:
```bash
pip install tensorflow torch torchvision torchaudio timm scikit-learn pandas numpy matplotlib streamlit
```

### Step 3: Run the Jupyter Notebook
Execute the transformer models using:
```bash
jupyter notebook
```
Open and run `train_using_pretrained_model_image_classifier_B16_Adagrad.ipynb` to train and evaluate the ViT model.

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

