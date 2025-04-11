<h1> My Kaggle notebook </h1>

Check out my project on Kaggle: [Avataar Project](https://www.kaggle.com/code/vijayj0shi/stable-diffusion-video-generation)

      (It I contains each versions I have saved for refrence)
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
The Stable Diffusion Inpainting model is loaded from stabilityai/stable-diffusion-2-inpainting. The model is configured to run on the GPU .
<br><br>


<img src="https://github.com/user-attachments/assets/1548b2a1-1be7-489c-8be7-64651498b8bf" alt="image"  height="300"/>
<br><br>

## 2. Download and Process Input Image
(This section defines a function to download the initial image from a URL and convert it into an RGB format using PIL.)

<br>

```python
def open_image(path):
    image= PIL.Image.open(path).convert("RGB")
    image=image.resize((512,512))
    return image

img_path =r'/kaggle/input/avataar-example/Avataar_example/example2.jpg'
init_image = open_image(img_path)
init_image
```


<img src="https://github.com/user-attachments/assets/2165596f-eae8-45a1-886d-e9249528c8ce" alt="image" width="500" height="300"/>
<br><br>

## 3. Create Object Mask for the Image

<br>

This project uses code or references from the following repository:

- **DIS: Deep Image Smoothing**  
  GitHub Repository: [xuebinqin/DIS](https://github.com/xuebinqin/DIS/tree/main)  
  Authors: Xuebin Qin, et al.  
  Link: [https://github.com/xuebinqin/DIS/tree/main](https://github.com/xuebinqin/DIS/tree/main)
  
  Description: The DIS project implements state-of-the-art deep learning methods for image smoothing and is used in various tasks like background smoothing and segmentation.

The object mask is created to ensure that the part of the image you want to keep unchanged is preserved. The mask identifies the object of interest by isolating it from a white background.
#### Example Usage:
```python


import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download official weights
if not os.path.exists("./saved_models"):
    !mkdir ./saved_models
    MODEL_PATH_URL = r"https://drive.google.com/uc?id=1KjdN6oNpt7pKzn1DPQ7d0PtGEY0v8EYz"
    gdown.download(MODEL_PATH_URL, "./saved_models/isnet.pth", use_cookies=False)


```

<br><br>

<img src="https://github.com/user-attachments/assets/51ec0fb0-4ab6-4adc-a231-a1f92d7ece8b" alt="image" width="500" height="300"/>
<br><br>


## 4. Apply Image Inpainting Using Depth Map and  Masked Image

<br>
Here, image inpainting is applied using a prompt that defines the modifications to the scene, along with the generated object mask and depth map. Various hyperparameters like num_inference_steps and guidance_scale control the generation process for creative freedom.
<br><br>

```python
from PIL import Image
prompt ="A modern   on road leaning against an old brown brick wall in natural sunlight ,real   with black shade borders,harmonize, add depth using depth map ,smooth tires. The scene is calm and realistic with subtle shadows, and soft lighting. No other objects around, focusing entirely on the bike use depth map to harmonize the   with background"
img_url='/kaggle/input/avataar-example/Avataar_example/example2.jpg'
init_image = open_image(img_path)
masked_image=create_object_mask(init_image)
depth_map_image=generate_depth_map(init_image)
# prompt="kitchen toaster , Studio quality picture ,inside kitchen"
negative_prompt=", draw cycle,No harsh lighting, no clutter no reflections, no unrealistic colors, no futuristic elements, no distorted shapes, no blurry objects, no floating items, no extreme highlights, no over-saturated colors, no chaotic composition,, no excessive details,do not change   do not add extra part on    ."

num_inference_steps = 500  # Refined image generation steps
guidance_scale = 2.5      # Balance between creativity and prompt adherence
# strength = 0.9            # Influence of the original image

# Generate inpainted image
new_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=masked_image,
    depth_map=depth_map_image,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
#     strength=strength,
    height=512,
    width=512
).images[0]
new_image

```


<img src="https://github.com/user-attachments/assets/5a7db679-8f27-4548-8b24-60d434678455" alt="image" width="500" height="300"/>
<br><br>



<br><br>
<img src="https://github.com/user-attachments/assets/e2fa9c73-2afa-43f1-98d1-0a48294f25ea" alt="image" width="500" height="300"/>
<br><br>


## 5. Generate Video Frames Based on Text Prompts
Video frames are generated for each prompt, creating a sequence of frames where the object remains unaltered while the scene evolves according to each prompt. The prompts are designed to create diverse scenes, and the depth map ensures realistic lighting and perspective adjustments.
<br><br>

```python
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Updated function to handle multiple prompts
def frame_creation(init_image, prompts, negative_prompt, scale_factor=0.77):
    frames = []
    guidance_scale = 8.5     
    strength = 0.9   
    inference_steps=20
    for prompt in prompts:
        # 1. First generate the image at original size for the current prompt
        scaled_image = init_image
        
        if isinstance(scaled_image, np.ndarray):
            scaled_image = Image.fromarray(scaled_image)

        # Apply inpainting with the original size for the current prompt
        mask_image_resized = create_object_mask(scaled_image)
        depth_map_resized=generate_depth_map(scaled_image)
        new_image_original = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image=scaled_image, 
            depth_map=depth_map_resized,
            mask_image=mask_image_resized, 
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
#             strength=strength,
            height=512,
            width=512
        ).images[0]
        
        # Append the original size frame
        frames.append(cv2.cvtColor(np.array(new_image_original), cv2.COLOR_RGB2BGR))

        # 2. Now resize the image by the scale factor (e.g., 0.77)
        scaled_image = resize_and_center_images(init_image, scale_factor)

        if isinstance(scaled_image, np.ndarray):
            scaled_image = Image.fromarray(scaled_image)

        # Apply inpainting with the scaled image for the same prompt
        mask_image_resized = create_object_mask(scaled_image)
        depth_map_resized=generate_depth_map(scaled_image)
        new_image_resized = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image=scaled_image,
            depth_map=depth_map_resized,             
            mask_image=mask_image_resized, 
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
#            strength=strength
            height=512,
            width=512
        ).images[0]
        
        # Append the resized frame
        frames.append(cv2.cvtColor(np.array(new_image_resized), cv2.COLOR_RGB2BGR))
    
    return frames


```
<img src="https://github.com/user-attachments/assets/d49ee0ae-2a0e-42a4-97f4-5266b15efbb1" alt="image" width="500" height="300"/>
<br><br>
<img src="https://github.com/user-attachments/assets/af7cabba-7a1a-48fe-a84f-2871eb77c44c" alt="image" width="500" height="300"/>
<br><br>



**Run the Code You can run the provided code either in a local environment or using a cloud platform like Google Colab or Kaggle. Make sure to have GPU access enabled.**

# Output

***The final output is a video showcasing the transitions between various prompts, with the object (e.g., a chair, electric cooker" 


<img src="https://github.com/user-attachments/assets/49d6431b-3e17-4c56-a884-fb7e4fa923f0" alt="image" />

***1. Customizing the Project***
<br>
***2. Change the Input Image***
<br>
***3. Replace the URL or path to the input image to work with different objects.***
<br>
***4. Modify Prompts***
<br>
***5. The prompts used for image generation can be customized to describe different scenes and objects.***
<br>
***6. Tweak Hyperparameters***
<br>
***7. Experiment with num_inference_steps, guidance_scale, and other parameters to see how they affect the quality and speed of image generation.***
<br>
# Conclusion
**This project demonstrates the powerful combination of Stable Diffusion, depth maps, and image inpainting for creating realistic, contextually rich images and videos. It showcases how AI models can be used to manipulate and enhance visuals while keeping specific elements intact.**
