import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from annoy import AnnoyIndex
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved models
pca = joblib.load('pca_model.pkl') # loaded from saved files
annoy_index = AnnoyIndex(256, 'euclidean')
annoy_index.load('annoy_index.ann')  # loaded from saved files
image_files = joblib.load('image_files.pkl')# loaded from saved files


# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS) # Use Image.LANCZOS instead of Image.ANTIALIAS
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

st.set_page_config(page_title="Image Matching")
# Title and description
st.title("Image Matching with Database")
st.caption("Powered by PIL, scikit-learn")

user_image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if user_image_path is not None:
    user_image = preprocess_image(user_image_path)
    user_image = np.expand_dims(user_image, axis=0)
    user_feature = base_model.predict(user_image)[0]
    user_feature_reduced = pca.transform([user_feature])[0]

    # Get the nearest neighbors
    num_neighbors = 5
    nearest_indices = annoy_index.get_nns_by_vector(user_feature_reduced, num_neighbors, include_distances=False)
    similar_images = [image_files[idx] for idx in nearest_indices]

    #comparison by confusion matrics
    similarities = cosine_similarity([user_feature_reduced], [pca.transform([base_model.predict(np.expand_dims(preprocess_image(path), axis=0))[0]])[0] for path in similar_images])[0]

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(user_image_path, width=300)

    # Display similar images and their similarity scores
    st.subheader("Top 5 Similar Images (evalution metrics: cosine similarity)")
    for i, (path, similarity) in enumerate(zip(similar_images, similarities)):
        st.write(f"Image {i+1} - Similarity: {similarity:.2f}")
        st.image(Image.open(path), width=150)
