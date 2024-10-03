My Kaggle notebook =   https://www.kaggle.com/code/vijayj0shi/avataar-projeccth1
      (It I contains each versions I have saved to verify I own the work)

<h1>Stable Diffusion Inpainting with Depth Maps for Video Generation</h1>

# Stable Diffusion Inpainting with Depth Maps for Video Generation

## Overview
This project focuses on image inpainting using the Stable Diffusion model, incorporating depth maps for realistic rendering. The main objective is to take an initial image and modify it based on text prompts while maintaining specific objects in the image unchanged. The project further extends to generate video frames, producing a natural, zoomed-out, or translated video output based on the prompts.

## Project Structure
1. **Load the Stable Diffusion Inpainting Pipeline**  
2. **Download and Process Input Image**  
3. **Create Object Mask for the Image**  
4. **Generate Depth Map Using Pretrained Model**  
5. **Apply Image Inpainting Using Depth Map**  
6. **Generate Video Frames Based on Text Prompts**  
7. **Save and Export Frames into a Video File**

## Step-by-Step Explanation
***1. Load the Stable Diffusion Inpainting Pipeline***
The Stable Diffusion Inpainting model is loaded from stabilityai/stable-diffusion-2-inpainting. The model is configured to run on the GPU (using CUDA) for faster computation.

![image](https://github.com/user-attachments/assets/edcc2f4b-3764-4efc-a953-0a98968121cd)


## 2. Download and Process Input Image
This section defines a function to download the initial image from a URL and convert it into an RGB format using PIL.

![image](https://github.com/user-attachments/assets/cc87606a-8239-4016-ab46-d96c37372f17)

![image](https://github.com/user-attachments/assets/e3854bb8-ec85-43da-9a9a-72a2f2434797)

## 3. Create Object Mask for the Image
The object mask is created to ensure that the part of the image you want to keep unchanged is preserved. The mask identifies the object of interest by isolating it from a white background.


![image](https://github.com/user-attachments/assets/e99a2cc0-8671-44a4-988e-7fff55869988)

![image](https://github.com/user-attachments/assets/64c1a40f-cca6-4a7c-b06c-b796ba4a2337)


## 4. Generate Depth Map Using Pretrained Model
A depth map is generated using a pretrained depth estimation model (Intel/dpt-large). This helps create a more realistic background based on the perspective and spatial features of the image.

![image](https://github.com/user-attachments/assets/59997235-7899-41d3-a623-fa7601984162)

![image](https://github.com/user-attachments/assets/a0baa8da-2211-4e4e-960b-89aa4ff7456b)

## 5. Apply Image Inpainting Using Depth Map
Here, image inpainting is applied using a prompt that defines the modifications to the scene, along with the generated object mask and depth map. Various hyperparameters like num_inference_steps and guidance_scale control the generation process for creative freedom.

![image](https://github.com/user-attachments/assets/68de1bc4-c6bf-491e-be3c-e5a66316e06e)

![image](https://github.com/user-attachments/assets/98602a3d-ffe5-4aab-be19-13e625d1ee21)

## 6. Generate Video Frames Based on Text Prompts
Video frames are generated for each prompt, creating a sequence of frames where the object remains unaltered while the scene evolves according to each prompt. The prompts are designed to create diverse scenes, and the depth map ensures realistic lighting and perspective adjustments.

python
Copy code
def frame_creation(init_image, prompts, negative_prompt, scale_factor=0.77):
    # Code to generate frames...
    return frames
Here, you can upload a gallery of generated frames, showing how the image changes with each prompt.

7. Save and Export Frames into a Video File
The frames are saved as a video file using OpenCV. You can adjust the frame rate (fps) and target size to achieve the desired output quality.

![image](https://github.com/user-attachments/assets/af7cabba-7a1a-48fe-a84f-2871eb77c44c)
![image](https://github.com/user-attachments/assets/b7885d27-0e86-4a00-9fe8-6fef9e4db54e)

![image](https://github.com/user-attachments/assets/34e0f68a-128e-4dc4-aec5-3f36ac91e9ef)

# How to Run the Project
**Install Dependencies**
**Ensure you have the following Python libraries installed:**
![image](https://github.com/user-attachments/assets/76fbe074-b047-49d6-b513-38efc0f06a2f)

**pip install diffusers transformers accelerate torch PIL numpy requests opencv-python matplotlib**
**Run the Code You can run the provided code either in a local environment or using a cloud platform like Google Colab or Kaggle. Make sure to have GPU access enabled.**

# Output

***The final output is a video showcasing the transitions between various prompts, with the object (e.g., a chair, electric cooker) remaining unchanged.***
***Customizing the Project***
***Change the Input Image***
***Replace the URL or path to the input image to work with different objects.***
***Modify Prompts***
***The prompts used for image generation can be customized to describe different scenes and objects.***
***Tweak Hyperparameters***
***Experiment with num_inference_steps, guidance_scale, and other parameters to see how they affect the quality and speed of image generation.***

# Conclusion
**This project demonstrates the powerful combination of Stable Diffusion, depth maps, and image inpainting for creating realistic, contextually rich images and videos. It showcases how AI models can be used to manipulate and enhance visuals while keeping specific elements intact.**

# Feel free to customize the code and experiment with different objects and environments to create your unique outputs!

