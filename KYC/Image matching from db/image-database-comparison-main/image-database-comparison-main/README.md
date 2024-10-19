# Image Similarity Search with MobileNetV2

This project demonstrates how to perform image similarity search using the MobileNetV2 model for feature extraction. It finds the top 5 most similar images from a dataset based on a user-provided image. The dataset used for this project is the ImageNet-mini dataset from Kaggle, and the similarity search is performed using PCA for dimensionality reduction, k-means clustering, and the Annoy library for nearest neighbor search.

## Features

- **Feature Extraction**: Uses MobileNetV2 to extract image features.
- **Dimensionality Reduction**: Reduces the high-dimensional features using PCA.
- **Clustering**: Groups similar features using k-means clustering.
- **Similarity Search**: Finds the top 5 most similar images using Annoy.
- **Deployment**: Can be deployed as a web app using Streamlit.

## Prerequisites

- Python 3.10 (or lower version)
- The following Python packages:
  - `tensorflow`
  - `numpy`
  - `pillow`
  - `scikit-learn`
  - `annoy`
  - `matplotlib`
  - `tqdm`
  - `joblib`
  - `streamlit`

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/gautam-17-2003/image-similarity-search.git
   cd image-similarity-search
   ```

2. **Install the Dependencies**
   ```bash
   pip install -r requirement.txt
   ```
3. **Further Step to proceed**
   - either you can upload the kaggle token from [here](https://www.kaggle.com/settings) or you can download the dataset manually from [here](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
   - run the `Imagemini_mobilenetv2.ipynb` file cell by cell
   - this will make the indexes and maps them with the images
   - after the mapping is done, run the `st_app.py` by running
     ```bash
     streamlit run st_app.py
     ```

## Code walkthrough
1. Feature Extraction: Uses MobileNetV2 to obtain feature vectors for images.
2. Dimensionality Reduction: PCA is used to reduce feature vector dimensionality to 256 dimensions.
3. Clustering: MiniBatchKMeans clusters the reduced feature vectors.
4. Nearest Neighbor Search: Annoy is used to find the nearest neighbors in the reduced feature space.
5. Integrating with streamlit: The app takes a user-provided image, finds the top 5 similar images, and displays them.

## Acknowledgments
- The ImageNet-mini dataset is used for educational purposes.
- MobileNetV2 is a pre-trained model available through TensorFlow

