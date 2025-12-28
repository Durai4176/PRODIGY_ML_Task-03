# 📊 Internship Task 3 – Streamlit Image Classification App (SVM)

## 📌 Project Overview
This project is a **Streamlit-based web application** developed as part of an internship task.  
The application performs **image classification** to identify whether an image is a **Cat** or a **Dog** using a **Support Vector Machine (SVM)** model.

The model is trained using a **training image dataset** and evaluated using a **testing image dataset**, both stored in folders.

---

## 🎯 Objective
- To classify images as **Cat or Dog** using **SVM**
- To understand how image data is read from folders
- To analyze the effect of dataset size on training time
- To build a simple and interactive Streamlit application

---

## 🛠️ Technologies Used
- Python  
- Streamlit  
- NumPy  
- Pillow (PIL)  
- Scikit-learn  

---

## 📂 Project Structure
```
PRODIGY_ML_Task-03/
│
├── app.py
├── README.md
└── dataset/
    ├── training_set/
    │   ├── cats/
    │   └── dogs/
    └── test_set/
        ├── cats/
        └── dogs/
```

---

## 📥 Dataset Information
- The dataset consists of **image files only** (no CSV file)
- Images are organized into folders
- Folder names act as **class labels**

## 📥 Dataset drive link:
```
https://drive.google.com/drive/folders/1uU5xeBckwdLph5aDehBqd5v47RKcTviO?usp=sharing

```


### Label Encoding
- cats → Label **0**  
- dogs → Label **1**

---

## 🧠 How the Code Takes Images from the Folder

### Step 1: Folder Path Detection
- The code automatically detects the dataset location using the position of `app.py`
- No manual file upload or path input is required

### Step 2: Reading Images
- The code scans the `cats` and `dogs` folders inside `training_set` and `test_set`
- Only image files (`.jpg`, `.png`, `.jpeg`) are considered

### Step 3: Image Preprocessing
Each image is:
1. Opened using the PIL library  
2. Converted to RGB format  
3. Resized to **64 × 64 pixels**  
4. Converted into a NumPy array  
5. Flattened into numerical values so that the SVM model can process it  

### Step 4: Label Assignment
- Images inside the `cats` folder are assigned label **0**
- Images inside the `dogs` folder are assigned label **1**

### Step 5: Training and Testing
- Images from the `training_set` folder are used to **train the SVM model**
- Images from the `test_set` folder are used to **test the model and calculate accuracy**

---

## ⏱️ Training Time vs Number of Images (Important Note)
- As the **number of training images increases**, the **model takes more time to train**
- This is because each image is converted into thousands of numerical features
- Using fewer images results in faster training but may reduce accuracy
- Using more images improves learning but increases training time

---

## 🚀 Features
- Automatic image loading from folders  
- Image preprocessing and feature extraction  
- SVM model training  
- Accuracy calculation using test data  
- Single image prediction  
- Simple and interactive Streamlit interface  

---

## ⚙️ Installation Steps

### Step 1: Install Python
Download Python from:  
https://www.python.org/

---

### Step 2: Install Required Libraries
```
pip install streamlit numpy pillow scikit-learn
```

---

### Step 3: Run the Application
```
streamlit run app.py
```

---

## 📊 Output
- Classification of images as **Cat or Dog**
- Model accuracy displayed in percentage
- Visual display of selected images and prediction results

---

## 🧾 Conclusion
This project demonstrates a basic **image classification system** using **Support Vector Machine (SVM)**.  
It explains how images are loaded from folders, processed, and used to train and evaluate the model.  
It also highlights how increasing the dataset size affects **training time and performance**.

---

## 👤 Author
Name: DURAIMURUGAN  
Project Type: Internship Task – Machine Learning  
Framework: Streamlit  
