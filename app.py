import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Cats vs Dogs Classification - SVM",
    page_icon="🐾",
    layout="wide"
)

# Title and Header
st.title("🐾 Cats vs Dogs Image Classification using SVM")
st.markdown("---")
st.markdown("""
This application uses a Support Vector Machine (SVM) classifier to distinguish between cat and dog images.
The model is trained on a training dataset and evaluated on a separate testing dataset.
""")

# Dataset paths
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "dataset" / "training_set"
TEST_DIR = BASE_DIR / "dataset" / "test_set"
TRAIN_CATS_DIR = TRAIN_DIR / "cats"
TRAIN_DOGS_DIR = TRAIN_DIR / "dogs"
TEST_CATS_DIR = TEST_DIR / "cats"
TEST_DOGS_DIR = TEST_DIR / "dogs"

# Image size for resizing
IMAGE_SIZE = (64, 64)

@st.cache_data
def load_images_from_folder(folder_path, label, max_images=None):
    """
    Load images from a folder and return as arrays with labels.
    
    Args:
        folder_path: Path to the folder containing images
        label: Label for the images (0 for cats, 1 for dogs)
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        images: List of image arrays
        labels: List of labels
    """
    images = []
    labels = []
    
    folder_path_str = str(folder_path)
    if not os.path.exists(folder_path_str):
        return images, labels
    
    image_files = [f for f in os.listdir(folder_path_str) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for img_file in image_files:
        try:
            img_path = os.path.join(folder_path_str, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img).flatten()  # Flatten to 1D array
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            st.warning(f"Error loading {img_file}: {str(e)}")
            continue
    
    return images, labels

@st.cache_resource
def train_svm_model(X_train, y_train):
    """
    Train an SVM classifier on the training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained SVM model
    """
    # Use a linear kernel for faster training on large datasets
    # You can experiment with 'rbf' kernel for potentially better accuracy
    clf = svm.SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    return clf

def main():
    st.header("📊 Model Training and Evaluation")
    
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Options")
        max_train_images = st.number_input(
            "Max training images per class",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Limit the number of training images per class for faster training"
        )
        max_test_images = st.number_input(
            "Max test images per class",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Limit the number of test images per class for faster evaluation"
        )
    
    # Load training data
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Loading training images..."):
            # Load cat images (label = 0)
            cat_images_train, cat_labels_train = load_images_from_folder(
                TRAIN_CATS_DIR, label=0, max_images=max_train_images
            )
            
            # Load dog images (label = 1)
            dog_images_train, dog_labels_train = load_images_from_folder(
                TRAIN_DOGS_DIR, label=1, max_images=max_train_images
            )
            
            if len(cat_images_train) == 0 or len(dog_images_train) == 0:
                st.error("❌ Error: Could not load training images. Please check the dataset paths.")
                return
            
            # Combine training data
            X_train = np.array(cat_images_train + dog_images_train)
            y_train = np.array(cat_labels_train + dog_labels_train)
            
            st.success(f"✅ Loaded {len(cat_images_train)} cat images and {len(dog_images_train)} dog images for training")
            st.info(f"📦 Training data shape: {X_train.shape}")
        
        # Train the model
        with st.spinner("Training SVM model... This may take a few minutes..."):
            model = train_svm_model(X_train, y_train)
            st.session_state['model'] = model
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
        
        st.success("✅ Model trained successfully!")
        
        # Load and evaluate on test data
        with st.spinner("Loading test images..."):
            # Load test cat images
            cat_images_test, cat_labels_test = load_images_from_folder(
                TEST_CATS_DIR, label=0, max_images=max_test_images
            )
            
            # Load test dog images
            dog_images_test, dog_labels_test = load_images_from_folder(
                TEST_DOGS_DIR, label=1, max_images=max_test_images
            )
            
            if len(cat_images_test) == 0 or len(dog_images_test) == 0:
                st.error("❌ Error: Could not load test images. Please check the dataset paths.")
                return
            
            # Combine test data
            X_test = np.array(cat_images_test + dog_images_test)
            y_test = np.array(cat_labels_test + dog_labels_test)
            
            st.success(f"✅ Loaded {len(cat_images_test)} cat images and {len(dog_images_test)} dog images for testing")
        
        # Evaluate the model
        with st.spinner("Evaluating model on test data..."):
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['accuracy'] = accuracy
            st.session_state['test_image_files'] = {
                'cats': [f for f in os.listdir(str(TEST_CATS_DIR)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_test_images],
                'dogs': [f for f in os.listdir(str(TEST_DOGS_DIR)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_test_images]
            }
        
        # Display accuracy
        st.markdown("---")
        st.header("📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        
        with col2:
            correct = np.sum(y_pred == y_test)
            st.metric("Correct Predictions", f"{correct}/{len(y_test)}")
        
        with col3:
            incorrect = np.sum(y_pred != y_test)
            st.metric("Incorrect Predictions", incorrect)
        
        # Confusion matrix-like display
        st.subheader("📊 Prediction Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cat Images (Label = 0)**")
            cat_test_count = len(cat_images_test)
            cat_correct = np.sum((y_pred[:cat_test_count] == 0))
            st.write(f"Correct: {cat_correct}/{cat_test_count}")
            st.write(f"Accuracy: {cat_correct/cat_test_count*100:.2f}%")
        
        with col2:
            st.write("**Dog Images (Label = 1)**")
            dog_test_count = len(dog_images_test)
            dog_correct = np.sum((y_pred[cat_test_count:] == 1))
            st.write(f"Correct: {dog_correct}/{dog_test_count}")
            st.write(f"Accuracy: {dog_correct/dog_test_count*100:.2f}%")
    
    # Single image prediction section
    st.markdown("---")
    st.header("🔍 Single Image Prediction")
    
    if 'model' in st.session_state:
        st.write("Select an image from the test dataset to predict:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            animal_type = st.selectbox("Select animal type", ["cats", "dogs"])
        
        if 'test_image_files' in st.session_state:
            available_images = st.session_state['test_image_files'].get(animal_type, [])
            
            if available_images:
                with col2:
                    selected_image = st.selectbox("Select image file", available_images)
                
                if st.button("🔮 Predict", type="primary"):
                    # Determine the folder path
                    if animal_type == "cats":
                        img_path = TEST_CATS_DIR / selected_image
                        true_label = 0
                    else:
                        img_path = TEST_DOGS_DIR / selected_image
                        true_label = 1
                    
                    # Load and preprocess the image
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_resized = img.resize(IMAGE_SIZE)
                        img_array = np.array(img_resized).flatten().reshape(1, -1)
                        
                        # Predict
                        prediction = st.session_state['model'].predict(img_array)[0]
                        prediction_prob = "Cat" if prediction == 0 else "Dog"
                        true_label_str = "Cat" if true_label == 0 else "Dog"
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📸 Selected Image")
                            st.image(img, width=300)
                        
                        with col2:
                            st.subheader("🎯 Prediction Results")
                            st.write(f"**Predicted Label:** {prediction_prob}")
                            st.write(f"**True Label:** {true_label_str}")
                            
                            if prediction == true_label:
                                st.success("✅ Correct Prediction!")
                            else:
                                st.error("❌ Incorrect Prediction")
                            
                            st.write(f"**Prediction Code:** {prediction} (0=Cat, 1=Dog)")
                    except Exception as e:
                        st.error(f"Error loading or predicting image: {str(e)}")
            else:
                st.warning("No images available for prediction. Please train the model first.")
        else:
            st.info("👆 Please train the model first to enable image prediction.")
    else:
        st.info("👆 Please click the 'Train Model' button above to start training the SVM classifier.")
    
    # Additional Information Section
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### How it Works:
        1. **Image Loading**: Images are loaded from the training and testing folders
        2. **Preprocessing**: All images are resized to 64×64 pixels and converted to RGB format
        3. **Feature Extraction**: Images are flattened into 1D arrays (12,288 features per image)
        4. **Training**: An SVM classifier with linear kernel is trained on the training data
        5. **Evaluation**: The model is evaluated on the test dataset to calculate accuracy
        6. **Prediction**: Individual images can be selected from the test set for prediction
        
        ### Label Encoding:
        - **Cats** → Label: 0
        - **Dogs** → Label: 1
        
        ### Model Details:
        - **Algorithm**: Support Vector Machine (SVM)
        - **Kernel**: Linear (for faster training)
        - **Image Size**: 64×64 pixels
        - **Features**: 64 × 64 × 3 = 12,288 features per image
        """)

if __name__ == "__main__":
    main()
