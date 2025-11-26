import streamlit as st
import os
import torch
from transformers import pipeline
from PIL import Image
import boto3

# -----------------------------
# AWS S3 SETTINGS
# -----------------------------
bucket_name = "agnishpaul"
local_path = "vit-human-pose-classification"

s3_prefix = "ml-models/vit-human-pose-classification/"

s3 = boto3.client("s3")

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for obj in result["Contents"]:
                s3_key = obj["Key"]

                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("ViT Human Pose Classification 🤸‍♂️")

if st.button("Download Model"):
    with st.spinner("Downloading model from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Model download complete!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):

    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "image-classification",
        model="./vit-human-pose-classification",
        device=device
    )

    with st.spinner("Predicting..."):
        output = classifier(image)
        st.write(output)
