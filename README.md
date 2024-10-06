**My Kaggle notebook

Check out my project on Kaggle: [Avataar Project](https://www.kaggle.com/code/vijayj0shi/avataar-projeccth1/edit/run/199673023)

      (It I contains each versions I have saved for refrence)**
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
The Stable Diffusion Inpainting model is loaded from stabilityai/stable-diffusion-2-inpainting. The model is configured to run on the GPU (using CUDA" alt="image" width="500" height="300"/> for faster computation.
<br><br>

<img src="https://github.com/user-attachments/assets/3dda4553-876d-4759-b820-4fa3c14d8300" alt="image" width="500" height="300"/>
<br><br>

## 2. Download and Process Input Image
This section defines a function to download the initial image from a URL and convert it into an RGB format using PIL.
<br><br>
<img src="https://github.com/user-attachments/assets/cc87606a-8239-4016-ab46-d96c37372f17" alt="image" width="500" height="300"/>
<br><br>

<img src="https://github.com/user-attachments/assets/2165596f-eae8-45a1-886d-e9249528c8ce" alt="image" width="500" height="300"/>
<br><br>
## 3. Create Object Mask for the Image
The object mask is created to ensure that the part of the image you want to keep unchanged is preserved. The mask identifies the object of interest by isolating it from a white background.

<br><br>

<img src="https://github.com/user-attachments/assets/51ec0fb0-4ab6-4adc-a231-a1f92d7ece8b" alt="image" width="500" height="300"/>
<br><br>

## 4. Generate Depth Map Using Pretrained Model
A depth map is generated using a pretrained depth estimation model (Intel/dpt-large" alt="image" width="500" height="300"/>. This helps create a more realistic background based on the perspective and spatial features of the image.
<br><br>
<img src="https://github.com/user-attachments/assets/59997235-7899-41d3-a623-fa7601984162" alt="image" width="500" height="300"/>
<br><br>
<img src="https://github.com/user-attachments/assets/c84467f4-7154-41e2-876a-683d63b37909" alt="image" width="500" height="300"/>
<br><br>
<img src="https://github.com/user-attachments/assets/07afe5d3-0706-415b-9c5f-0241316a4f32" alt="image" width="500" height="300"/>
<br><br>
## 5. Apply Image Inpainting Using Depth Map
Here, image inpainting is applied using a prompt that defines the modifications to the scene, along with the generated object mask and depth map. Various hyperparameters like num_inference_steps and guidance_scale control the generation process for creative freedom.
<br><br>
<img src="https://github.com/user-attachments/assets/5a7db679-8f27-4548-8b24-60d434678455" alt="image" width="500" height="300"/>
<br><br>



<br><br>
<img src="https://github.com/user-attachments/assets/98602a3d-ffe5-4aab-be19-13e625d1ee21" alt="image" width="500" height="300"/>
<br><br>


## 6. Generate Video Frames Based on Text Prompts
Video frames are generated for each prompt, creating a sequence of frames where the object remains unaltered while the scene evolves according to each prompt. The prompts are designed to create diverse scenes, and the depth map ensures realistic lighting and perspective adjustments.
<br><br>
<img src="https://github.com/user-attachments/assets/d49ee0ae-2a0e-42a4-97f4-5266b15efbb1" alt="image" width="500" height="300"/>
<br><br>
<img src="https://github.com/user-attachments/assets/af7cabba-7a1a-48fe-a84f-2871eb77c44c" alt="image" width="500" height="300"/>
<br><br>

# 7. Save and Export Frames into a Video File
The frames are saved as a video file using OpenCV. You can adjust the frame rate (fps" alt="image" width="500" height="300"/> and target size to achieve the desired output quality.
<br><br>
<img src="https://github.com/user-attachments/assets/b7885d27-0e86-4a00-9fe8-6fef9e4db54e" alt="image" width="500" height="300"/>
<br><br>

<img src="https://github.com/user-attachments/assets/34e0f68a-128e-4dc4-aec5-3f36ac91e9ef" alt="image" width="500" height="300"/>
<br><br>
# How to Run the Project
**Install Dependencies**
**Ensure you have the following Python libraries installed:**
<br><br>
<img src="https://github.com/user-attachments/assets/76fbe074-b047-49d6-b513-38efc0f06a2f" alt="image" />
<br><br>
**pip install diffusers transformers accelerate torch PIL numpy requests opencv-python matplotlib**
**Run the Code You can run the provided code either in a local environment or using a cloud platform like Google Colab or Kaggle. Make sure to have GPU access enabled.**

# Output

***The final output is a video showcasing the transitions between various prompts, with the object (e.g., a chair, electric cooker" alt="image" width="500" height="300"/> remaining unchanged.***
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

