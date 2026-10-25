import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Generative AI Image Caption Generator")
st.write(
    "Upload an image and use the pretrained BLIP model "
    "to generate a natural-language caption."
)


@st.cache_resource
def load_model():
    """Download and cache the pretrained BLIP model."""
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return processor, model


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Generate caption", type="primary"):
            with st.spinner(
                "Loading the AI model and generating a caption..."
            ):
                processor, model = load_model()
                inputs = processor(image, return_tensors="pt")

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=50,
                    )

                caption = processor.decode(
                    output[0],
                    skip_special_tokens=True,
                )

            st.subheader("Generated caption")
            st.success(caption)

    except Exception as exc:
        st.error(
            "Could not process this image or load the model. "
            "Check your internet connection and installed packages, "
            f"then try again. Details: {exc}"
        )
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image to get started.")

st.caption(
    "This demo uses a pretrained BLIP model. Captions can be incorrect; "
    "review the output before relying on it."
)
