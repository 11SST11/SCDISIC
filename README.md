# Skin Cancer Detection Using Convolutional Neural Networks (CNNs)

## 📜 Project Overview
Skin cancer is one of the most common forms of cancer worldwide. Early and accurate diagnosis is crucial for effective treatment and better patient outcomes. This project leverages **deep learning techniques**, specifically **Convolutional Neural Networks (CNNs)**, to classify skin lesions into nine categories using the **ISIC dataset**.

We developed three CNN models with incremental enhancements to improve performance. The final model achieved a validation accuracy of **88.96%**, showcasing its potential as a supplementary diagnostic tool for dermatologists.

---

## 🚀 Features
- **Multi-class Classification:** Classifies skin lesions into nine categories, including melanoma, basal cell carcinoma, and seborrheic keratosis.
- **Progressive Model Development:** Implemented three CNN models with increasing complexity:
  1. **Model 1:** Basic CNN.
  2. **Model 2:** CNN with Dropout and Data Augmentation.
  3. **Model 3:** Advanced CNN with Batch Normalization.
- **Data Augmentation:** Improves generalization by applying transformations such as random flips, rotations, zooms, and translations.
- **Class Imbalance Handling:** Used **Augmentor** to balance the dataset by generating synthetic samples for underrepresented classes.
- **Visualization:** Training/validation accuracy and loss graphs for detailed performance insights.

---

## 📊 Performance Summary
| **Model**   | **Architecture Enhancements**                               | **Training Accuracy (%)** | **Validation Accuracy (%)** | **Training Loss** | **Validation Loss** |
|-------------|------------------------------------------------------------|----------------------------|------------------------------|-------------------|---------------------|
| **Model 1** | Basic CNN (3 Conv Layers, No Dropout, No Data Augmentation) | 81.22                     | 46.09                       | 0.4950            | 2.5895              |
| **Model 2** | Dropout and Data Augmentation                               | 14.56                     | 14.03                       | 2.1763            | 2.1881              |
| **Model 3** | Batch Normalization, Dropout, Augmentation (Advanced CNN)   | 92.33                     | **88.96**                   | 0.2093            | 0.3255              |

---

## 🛠️ Methodology

### 1. **Data Preprocessing**
- Images were resized to **180x180** pixels.
- Rescaled pixel values to the range [0, 1] using `tf.keras.layers.Rescaling`.
- Handled class imbalance using **Augmentor** to generate additional samples for minority classes.

### 2. **Model Architecture**
#### **Model 3: Advanced CNN**
```plaintext
Input Layer: Rescaling (1./255)
  Conv Layer 1: 32 filters, 2x2 kernel, ReLU activation, MaxPooling2D
  Conv Layer 2: 64 filters, 2x2 kernel, ReLU activation, MaxPooling2D
  Conv Layer 3: 128 filters, 2x2 kernel, ReLU activation, MaxPooling2D
  Fully Connected Layers:
    Dense Layer 1: 512 neurons, ReLU activation, Dropout (25%)
    Batch Normalization
    Dense Layer 2: 128 neurons, ReLU activation, Dropout (50%)
    Batch Normalization
  Output Layer: Softmax activation for 9 classes
```

### 3. **Training Configuration**
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)

---

## 📈 Visualizations
### Training and Validation Accuracy
![Accuracy Plot](https://via.placeholder.com/800x400)

### Training and Validation Loss
![Loss Plot](https://via.placeholder.com/800x400)

---

## 🔑 Key Insights
1. **Incremental Improvements:**
   - Model 3 significantly outperformed Models 1 and 2, highlighting the importance of batch normalization and advanced dropout strategies.
   - Data augmentation and handling class imbalance improved the generalization capability of the model.

2. **Challenges:**
   - Misclassifications between visually similar lesions (e.g., basal cell carcinoma and squamous cell carcinoma).
   - Model 2 struggled with convergence despite the introduction of dropout and augmentation.

3. **Future Enhancements:**
   - Implementing hybrid models using architectures like **ResNet50** and **VGG19**.
   - Exploring multimodal data integration, such as combining clinical metadata with image data.

---

## 📂 Project Structure
```plaintext
.
├── data
│   ├── train/       # Training dataset
│   ├── val/         # Validation dataset
├── models
│   ├── model1.py    # Basic CNN
│   ├── model2.py    # CNN with Dropout and Augmentation
│   ├── model3.py    # Advanced CNN
├── notebooks
│   ├── preprocessing.ipynb  # Data preprocessing and augmentation
│   ├── training.ipynb       # Model training and evaluation
├── plots
│   ├── accuracy_plot.png
│   ├── loss_plot.png
├── README.md
```

---

## 🔬 Technologies Used
- **Python**
- **TensorFlow/Keras**
- **Augmentor** for data augmentation
- **Matplotlib** and **Seaborn** for visualizations
- **Pandas** and **NumPy** for data handling

---

## 📢 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/skin-cancer-detection.git
   cd skin-cancer-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from the [ISIC Archive](https://isic-archive.com) and place it in the `data/` directory.
4. Run the training script:
   ```bash
   python models/model3.py
   ```
5. View performance metrics and plots in the `plots/` directory.

---

## 🧑‍💻 Contributors
- [Your Name](https://github.com/username)  

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgments
- **ISIC Archive** for the dataset.
- TensorFlow/Keras documentation for guiding the model implementation.

---

### 🌟 If you find this project helpful, give it a ⭐ on GitHub!
