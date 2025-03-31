# app.py
import streamlit as st
from PIL import Image
import classifier # Import your classifier functions

# --- Streamlit Configuration ---
st.set_page_config(page_title="Vehicle Classifier", layout="centered")

# --- Model Loading ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def get_model():
    """Loads and caches the classification model."""
    return classifier.load_model()

model = get_model()

# --- App Title and Description ---
st.title("🚗 B-C-B Classifier 🚲")
st.markdown("""
Upload an image of a **Bicycle**, **Car**, or **Bus/Truck**, and the AI will try to classify it!
This uses a pre-trained model (MobileNetV2) on ImageNet.
""")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Classification Logic ---
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Add a placeholder for status updates
    status_text = st.empty()
    status_text.text("Classifying...")

    # Perform classification
    # Pass the uploaded file object directly to the classifier function
    result = classifier.classify_image(model, uploaded_file)

    # --- Display Results ---
    if result:
        category, score, top_5 = result
        status_text.text("Classification Complete!") # Update status

        st.subheader("Classification Result:")

        if category != "Other":
            st.success(f"✅ This looks like a **{category}**! (Confidence: {score:.2f})")
        else:
            st.warning("⚠️ Could not confidently classify as Bicycle, Car, or Bus/Truck.")
            st.info("It might be something else, or the object wasn't clear in the image.")

        # Optionally display the top 5 raw predictions
        with st.expander("Show Top 5 Raw Predictions (from Model)"):
            st.write(top_5)
    else:
        status_text.error("❌ Error during classification. Please try another image.")

else:
    st.info("👆 Upload an image to get started.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io) and [TensorFlow/Keras](https://www.tensorflow.org).")
