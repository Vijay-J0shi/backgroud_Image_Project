<h1> My Kaggle notebook </h1>

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
The Stable Diffusion Inpainting model is loaded from stabilityai/stable-diffusion-2-inpainting. The model is configured to run on the GPU .
<br><br>

<img src="https://github.com/user-attachments/assets/3dda4553-876d-4759-b820-4fa3c14d8300" alt="image" width="500" height="300"/>
<br><br>

## 2. Download and Process Input Image
This section defines a function to download the initial image from a URL and convert it into an RGB format using PIL.

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
## Object Mask Creation

This project includes functionality for creating an object mask from an image using the IS-Net model. The function `create_object_mask` processes an input image to generate an inverted mask and returns a grayscale version of the masked image.

### Function: `create_object_mask(image)`

#### Parameters:
- **image**: A PIL Image object representing the input image from which the object mask will be created.

#### Description:
1. **Load the Image**: The function converts the input image to a NumPy array and loads it into a tensor format suitable for the neural network model.
2. **Predict Mask**: It uses a pre-trained model (`net`) to predict the mask of the object within the image.
3. **Inversion and Resizing**: The predicted mask is inverted, resized to match the original image size, and converted to a grayscale image.
4. **Return Value**: The function returns a grayscale image of the inverted mask.

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



...
....
....
...

    
def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

  
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need



def create_object_mask(image):
    cv_image = np.array(image)
    image_tensor, orig_size = load_image(cv_image, hypar) 
    mask = predict(net, image_tensor, orig_size, hypar, device)

    inverted_mask = np.max(mask) - mask
    inverted_mask_pil = Image.fromarray((inverted_mask * 255).astype(np.uint8))  # Scale to 255 for proper visualization

    init_image_size = (init_image.size[0], init_image.size[1]) 
    inverted_mask_resized = inverted_mask_pil.resize(init_image_size, Image.BILINEAR)  

    inverted_image = Image.eval(inverted_mask_resized, lambda x: 255 - x)

    # Convert the inverted image to 'L' mode (grayscale)
    inverted_image_l = inverted_image.convert('L')

    return inverted_image_l

# Example usage of create_object_mask
masked_image = create_object_mask(init_image)
```

<br><br>

<img src="https://github.com/user-attachments/assets/51ec0fb0-4ab6-4adc-a231-a1f92d7ece8b" alt="image" width="500" height="300"/>
<br><br>

## 4. Generate Depth Map Using Pretrained Model
A depth map is generated using a pretrained depth estimation model . This helps create a more realistic background based on the perspective and spatial features of the image.
<br><br>

This code leverages the `DPTForDepthEstimation` model from the Hugging Face `transformers` library to generate depth maps for images. The depth maps are then used to assist in the inpainting process, which is useful for tasks like realistic object blending or image manipulation.

## Code Explanation

### Libraries Used:
- **transformers**: Provides pre-trained models and tokenizers. In this case, we use the `DPTForDepthEstimation` model for depth map generation.
- **torch**: PyTorch is used for handling tensors and leveraging GPU capabilities for faster computation.
- **numpy**: Used for numerical operations, including array manipulation.
- **PIL**: Python Imaging Library is used for handling image processing tasks like resizing and overlaying images.

### Key Concepts:
1. **Depth Estimation Model**:
   - The `DPTForDepthEstimation` model from the "Intel/dpt-large" pre-trained model is used to predict the depth map of the input image.
   - The `DPTFeatureExtractor` processes the input image into the correct format for the model.

2. **Depth Map Generation**:
   - The `generate_depth_map()` function takes in an image, preprocesses it, and passes it through the depth estimation model to predict the depth map.
   - The resulting depth map is resized to match the input image size and normalized for better visualization.
   - The depth map is then converted to a grayscale image using `PIL.Image`.

3. **Image Overlay with Mask**:
   - In the `depth_map_image_fun()` function, a mask image is overlaid onto the generated depth map.
   - The `mask_image` is resized to match the dimensions of the depth map, and then the overlay is performed, maintaining transparency if applicable.


```python
# Import necessary libraries
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Load the depth estimation model and feature extractor
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to("cuda")

# Function to generate depth map
def generate_depth_map(image):
    # Preprocess the input image
    inputs = feature_extractor(images=image, return_tensors="pt").to("cuda")
    
    # Generate the depth map using the model
    with torch.no_grad():
        depth_output = depth_model(**inputs).predicted_depth
    
    # Resize the depth map to match the input image size
    depth_map = torch.nn.functional.interpolate(
        depth_output.unsqueeze(1), 
        size=image.size[::-1],  # PIL image size is in (width, height), so reverse the dimensions
        mode="bicubic",
        align_corners=False
    ).squeeze()

    # Normalize the depth map for visualization and use
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Convert the depth map to a PIL image for further processing
    depth_map_image = Image.fromarray((depth_map.cpu().numpy() * 255).astype(np.uint8))
    
    return depth_map_image


from PIL import Image

# Open the two images
def depth_map_image_fun(mask_image,init_image):
    depth_map_image = generate_depth_map(init_image)
    background = depth_map_image
    overlay = mask_image

    # Ensure the overlay image is the same size as the background (optional)
    overlay = overlay.resize(background.size)

    # Paste the overlay image onto the background with transparency (if needed)
    background.paste(overlay, (0, 0), overlay)
    return background
depth_map=generate_depth_map(init_image)
depth_map

```


<img src="https://github.com/user-attachments/assets/07afe5d3-0706-415b-9c5f-0241316a4f32" alt="image" width="500" height="300"/>
<br><br>
## 5. Apply Image Inpainting Using Depth Map and Item Mask
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


## 6. Generate Video Frames Based on Text Prompts
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

# Generate frames for each prompt with scaling
frames = frame_creation(init_image, video_prompts, video_negative_prompt, scale_factor=0.77)


num_frames = len(frames)
cols = 4  # Number of columns
rows = (num_frames // cols) + (1 if num_frames % cols else 0)  # Determine rows based on frame count

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
axes = axes.flatten()

# Display the frames
for i, frame in enumerate(frames):  # Adjust the number of displayed frames based on your grid size
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR back to RGB for display
    axes[i].axis('off')
    axes[i].set_title(f'Frame {i+1}')

plt.tight_layout()
plt.show()

```
<img src="https://github.com/user-attachments/assets/d49ee0ae-2a0e-42a4-97f4-5266b15efbb1" alt="image" width="500" height="300"/>
<br><br>
<img src="https://github.com/user-attachments/assets/af7cabba-7a1a-48fe-a84f-2871eb77c44c" alt="image" width="500" height="300"/>
<br><br>

# 7. Save and Export Frames into a Video File

```python
import cv2
import numpy as np

def save_frames_to_video(frames, output_path='/kaggle/working/output_video.mp4', fps=1, video_size=(512, 512)):
    # Define the video codec and output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    for frame in frames:
        # Convert the frame to BGR format for OpenCV
        video_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Get the size of the frame
        h, w, _ = video_frame.shape

        # Calculate the average color of the frame
        avg_color_per_row = np.average(video_frame, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0).astype(int)
        avg_color = tuple(avg_color[::-1])  # Convert from BGR to RGB

        # Create a blank canvas with the target video size using the average color
        canvas = np.full((video_size[1], video_size[0], 3), avg_color, dtype=np.uint8)

        # Calculate the top-left corner to place the frame on the canvas to center it
        x_offset = (video_size[0] - w) // 2
        y_offset = (video_size[1] - h) // 2

        # Place the frame in the center of the canvas, keeping its original size
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = video_frame

        # Write the frame to the video
        video_writer.write(canvas)

    # Release the video writer
    video_writer.release()

    print(f"Video saved at {output_path}")

save_frames_to_video(frames, output_path='/kaggle/working/output_video.mp4')

```
The frames are saved as a video file using OpenCV. You can adjust the frame rate (fps" alt="image" width="500" height="300"/> and target size to achieve the desired output quality.
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

# Feel free to customize the code and experiment with different objects and environments to create your unique outputs!

