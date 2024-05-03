import streamlit as st
from PIL import Image
import numpy as np

# import cv2
# import torch
# import yaml
# import albumentations as albu
# from albumentations.core.composition import Compose
# import archs
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


cols1, cols2 = st.columns([1, 4])

with cols1:
    st.image('./logo.png')

with cols2:
    st.write("Rajkiya Engineering College Kannauj")

st.title("Brain Tumor Segmentation")
st.sidebar.image("./logo.png", width=150)
st.sidebar.write("Early Brain Tumor Detection System using Modified U-Net")

def segment_image(image_np, model_name):
    print("Hello")
    # Load model configuration
    # with open('models/%s/config.yml' % model_name, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # # Load model
    # model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'])
    # model.load_state_dict(torch.load('models/%s/model.pth' % model_name))

    # # Move model to appropriate device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model.eval()

    # # Convert image to RGB if it's in BGR
    # if image_np.shape[2] == 3:  # Check if the image is BGR
    #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # # Apply transformations
    # transform = Compose([
    #     albu.Resize(config['input_h'], config['input_w']),
    #     albu.Normalize(),
    # ])
    # transformed_image = transform(image=image_np)['image']
    # transformed_image = torch.unsqueeze(torch.from_numpy(transformed_image.transpose(2, 0, 1)), dim=0).float()

    # # Move input image to the appropriate device
    # transformed_image = transformed_image.to(device)

    # # Predict segmentation mask
    # with torch.no_grad():
    #     output = model(transformed_image)
    #     output = torch.sigmoid(output).cpu().numpy()

    # # Save segmented image
    # output_image = output[0].transpose(1, 2, 0)
    # output_image = (output_image * 255).astype('uint8')
    # segmented_image = Image.fromarray(output_image)

    # return segmented_image

def main():
    mri_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"], key=1)
    mask_file = st.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"], key=2)

    if mri_file is not None and mask_file is not None:
        mriImage = Image.open(mri_file)
        mri_np = np.array(mriImage)

        maskImage = Image.open(mask_file)

        col1, col2, col3 = st.columns(3)

        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        with col2:
            st.image(maskImage, caption='Uploaded Mask Image')

        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            with col3:
                st.image(maskImage, caption='Segmented Image')


    if mri_file is not None and mask_file is None:
        mriImage = Image.open(mri_file)
        mri_np = np.array(mriImage)

        col1, col2 = st.columns(2)
        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            with col2:
                st.image(mriImage, caption='Segmented Image')

if __name__ == '__main__':
    main()
