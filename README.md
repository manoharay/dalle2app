# DALLE2 Application

Welcome to the first version of the DALLE2 application! You can try the application [here]([link](https://dalle2inpainter.streamlit.app/)). With this application, you can take a picture of yourself, provide a prompt, and adjust settings to generate a resulting image.

## Installation

To install the app scripts, follow these steps:

1. Clone the repository to your local machine using the command:
   ```git clone https://github.com/pr0fi7/dalle2app.git```
2. Ensure that you have the required dependencies installed by executing:
  ```pip install -r requirements.txt```
3. Create a `.TOML` secrets file in the advanced settings of the Streamlit app with your OpenAI API key:
  ```OPENAI_API_KEY = 'sk-hghghghhhfhfhffhhf'```


# How the Program Works

The program operates in the following manner:

1. **Image Input**: The program takes an image input from the Streamlit interface.
   
2. **Image Resizing**: The input image is resized to a standardized size using an intelligent and straightforward approach. This resizing ensures consistency in processing.
   
3. **Face Detection**: OpenCV's Haar Cascade is employed to detect the position of the face in the resized image. This step is crucial for subsequent processing.
   
4. **Mask Creation**: Based on the detected face position, a mask is created. This mask is utilized for specific operations, such as inpainting or image generation.
   
5. **Prompt Integration**: Alongside the resized image and mask, a user-provided prompt is integrated into the process. This prompt guides the generation or inpainting process by providing context and constraints.
   
6. **API Interaction**: The resized image, mask, and prompt are passed to the DALLE2 API. This API performs the generation or inpainting based on the provided inputs.
   
7. **Result Display**: The DALLE2 API returns a link to the generated or inpainted image. This resulting image is then displayed to the user via the Streamlit interface.

This streamlined process ensures that users can easily interact with the application, providing inputs and receiving outputs seamlessly. The combination of intelligent resizing, face detection, prompt integration, and API interaction results in efficient and effective image generation or inpainting.

## Future Changes

For future versions, the following improvements are planned:

- **Image Generation**: Expand the functionality to not only inpaint images but also generate entirely new ones. This enhancement would offer users more creative possibilities and applications.
  
- **Improved Mask Generation**: Integrate an external API, such as MediaPipe, to generate better masks for inpainting. While this was not utilized in the current version due to GPU requirements, it could significantly enhance the quality of inpainted images.
  
![0_DTOGUpnCcCEpJiIr](https://github.com/pr0fi7/dalle2app/assets/53155116/59153639-1e3a-40f5-b718-f3b2b09788f2)

