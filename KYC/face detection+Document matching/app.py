import streamlit as st
import cv2
from PIL import Image
import numpy as np
from comparison import live_doc_comparison
import time
from live_cap_new import blink_and_tilt_detection

def main():
    st.set_page_config(page_title="Face Verification App")

    # Title and description
    st.title("Face Verification: Upload Image and Compare with Live Webcam")
    st.caption("Powered by OpenCV, Streamlit, and DeepFace")

    # Step 1: Upload an image (ID image)
    uploaded_id_image = st.file_uploader("Upload an ID card image", type=["jpg", "png", "jpeg"])

    if uploaded_id_image is not None:
        # Convert uploaded image to a format compatible with OpenCV
        uploaded_image = Image.open(uploaded_id_image)
        uploaded_image_np = np.array(uploaded_image)  # Convert to numpy array
        uploaded_image_cv2 = cv2.cvtColor(uploaded_image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for DeepFace

    # Initialize webcam and placeholders for image

    frame_placeholder = st.empty()
    verification_result_placeholder = st.empty()

    if st.button("Start Detection"):
        success, final_frame = blink_and_tilt_detection(frame_placeholder)

        if success:
            verification_result_placeholder.success("Live photo verified!")
        else:
            st.warning("Timeout: Not Live Image. Try again.")

        try:
            # Perform DeepFace verification between uploaded image and live webcam image
            print(type(final_frame),type(uploaded_image_cv2))
            verification_result = live_doc_comparison(final_frame,uploaded_image_cv2)
            if final_frame:
                st.image()
            # Check the verification result
            if verification_result:
                verification_result_placeholder.success("Face match with id successfully!")
            else:
                verification_result_placeholder.error("Face match failed!")
        except Exception as e:
            verification_result_placeholder.error(f"Verification Error: {e}")


if __name__ == "__main__":
    main()
