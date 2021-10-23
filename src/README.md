# LungSegmentation_Streamlit
CXR segmentation with **streamlit**

# Dataset
* Montgomery
* Shenzhen

https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

## Pre-Process
* Original (None)
* DHE

# model
### U-Net

![Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al](https://user-images.githubusercontent.com/73995528/133255424-8fc99c61-e163-4f27-bd24-2760102ee121.png)

### residual U-Net

![arch](https://user-images.githubusercontent.com/73995528/133255516-5b2183ac-ebdc-4795-b8fd-e2481793c0e9.png)

_pretrained weights are stored in weights folder_

# Usage
    chmod +x ./run.sh
or 

    streamlit run ./ui.py

Automatically open up this link
* Local URL: http://localhost:8502

# Results
## APP 
![App](https://user-images.githubusercontent.com/73995528/133257206-7e2f8ea7-b7c9-48e2-b7a0-8806052bdd4a.png)
------------------------------------------------------------------

## Output
![results](https://user-images.githubusercontent.com/73995528/133257338-38b94363-39ff-43fa-9d41-161344a644ce.png)
------------------------------------------------------------------

# Variational Autoencoders (VAE)
![Screenshot from 2021-09-15 16-21-59](https://user-images.githubusercontent.com/73995528/133428563-3f6eb493-824f-4498-a586-7eb6acdfc103.png)
--------------------------------------------------------------------

# Parser
    python service.py -h
| small flag  |  flag     | Description     |
| :----:      |    :----: |     :---:      |
|-h| --help|            show this help message and exit|
|-p | --path|           Path to Image [images/00000165_008.png]|
|-d| --dhe |            Select Pre-Process|
|-n | --n_samples|      Number of Output [1, as many as in folder path]|
|-S| --streamlit   |    Run App|
|-U| --unet       |     Select U-Net|
|-R| --resunet     |    Select Residual U-Net|
|-V| --vae          |   Select Variational Autoencoder|
|-vr | --vae_range   |  Autoencoder Range [1, 30]|
|-on | --output_number| Number of Output [1, 30]|

**all outputs stored in images directory**




# Dockerfile

    docker build -t <image-name> .
    docker run -p 8501:8501 -d <image-name>


https://github.com/pejmanS21/LungSegmentation_Streamlit.git