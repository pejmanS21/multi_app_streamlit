import torch
from torch import nn
from torchvision import models
from torchvision import transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp


import streamlit as st

import logging

logger = logging.getLogger ('classification_app')
#adding level to logger
logger.setLevel (logging.DEBUG)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/multi_app.log')
# adding formatter to file_handler
file_handler.setFormatter (formatter)

# adding file & stream handler to logger
logger.addHandler (file_handler)

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')


def app ():
    def Image_to_torch (file):
        """ convert image from disk to torch tensor """
        tfms = transforms.Compose ([
        transforms.Resize ((224,224)),
        transforms.ToTensor ()
        ])

        img = Image.open (file).convert ('RGB')
        torch_img = tfms (img)
        
        logger.debug ('Transform image to torch was done')
        return torch_img


    def normalize (torch_img):
        """ normalize tensor with imagenet mean and std """
        tfms1 = transforms.Normalize (
                        mean = [0.485, 0.456, 0.406], 
                        std = [0.229, 0.224, 0.225])
        norm_img = tfms1 (torch_img)[None]
        logger.debug ('Normalized')
        return norm_img


    def Tensor_to_array (img):
        """ gpu tensor => cpu array """
        return img.cpu ().numpy ().transpose (1, 2, 0)


    def load_checkpoint (checkpoint_path, model, device):
        """ loading model's weights """
        model.load_state_dict (torch.load (checkpoint_path, map_location = device) ['state_dict'])


    def get_model (model_name, device, checkpoint_path = None):
        """ select imagenet models by their name and loading weights """
        if checkpoint_path:
            pretrained = False
        else:
            pretrained = True
        
        model = models.__dict__ [model_name](pretrained)
        
        logger.debug ('model selected truely!')
        if hasattr (model, 'classifier'):
            if model_name == 'mobilenet_v2':
                model.classifier = nn.Sequential(
                    nn.Dropout (0.2),
                    nn.Linear (model.classifier [-1].in_features, 2))
                
            else:
                model.classifier = nn.Sequential(
                    nn.Linear (model.classifier.in_features, 2))
        
        elif hasattr (model, 'fc'):
            model.fc = nn.Linear (model.fc.in_features, 2)
            
        logger.debug ('top layer changed')
        model.to(device)
        
        if checkpoint_path:
            load_checkpoint (checkpoint_path, model, device)
            logger.debug ('weights loaded')
        
        return model


    def predict_and_gradcam (model, torch_img, device):
        """ predict class and plot image with their gradcam results """
        img = normalize (torch_img)
        out = model (img.to(device))
        logger.debug ('classification scores:{}'.format (out))
        # predict class
        _, pred = torch.max (out, dim = 1)
        cls2idx = ['Normal', 'Pneumomia']
        # ploting GradCam
        fig, ax = plt.subplots (nrows = 1, ncols = 5, figsize = (15, 5))
        
        # usage layer for gradcam
        if hasattr (model, 'fc'):
            target_layer = model.layer4 [2].bn3
            
        else:
            target_layer = model.features
    
        logger.debug ('target layer selected')
        
        gradcam = GradCAM (model, target_layer)
        gradcam_pp = GradCAMpp (model, target_layer)
        
        mask, _ = gradcam (img.to(device))
        # mask, _ = gradcam (img)
        heatmap, result = visualize_cam (mask, torch_img)

        mask_pp, _ = gradcam_pp (img.to(device))
        # mask_pp, _ = gradcam_pp (img)
        heatmap_pp, result_pp = visualize_cam (mask_pp, torch_img)

        ax [0].imshow (Tensor_to_array (img [0])[:, :, 0], cmap = 'gray')
        ax [0].set_title ('Predict: {}'.format (cls2idx [pred]))
        ax [0].axis ('off')

        ax [1].imshow (Tensor_to_array (heatmap))
        ax [1].set_title ('Grad Cam')
        ax [1].axis ('off')

        ax [2].imshow (Tensor_to_array (heatmap_pp))
        ax [2].set_title ('Grad Cam ++')
        ax [2].axis ('off')

        ax [3].imshow (Tensor_to_array (result))
        ax [3].set_title ('Result')
        ax [3].axis ('off')

        ax [4].imshow (Tensor_to_array (result_pp))
        ax [4].set_title ('Result ++')
        ax [4].axis ('off')
        plt.show ()
        
        logger.debug ('GradCam images ploted!')
        return fig


    def load_model (model_name = 'mobilenet_v2'):
        """ loading Classification model """
        return get_model (model_name, device, '../weights/' + model_name + '.pth')


    response_code = 100

    st.markdown("""
                # Classification 
                A Classification app with **MobileNet_V2** and `Pytorch` as framework. 
                Upload your CXR Image and hit the `Submit` button to get predictions.

                -------------------------------------------------------------------------    
    """)
    with st.form(key='Classification'):
        with st.sidebar:
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                
                if file is not None:
                    # convert PIL image to torch image
                    image = Image_to_torch (file)
                    # convert torch to array to show
                    st.image(Tensor_to_array (image), use_column_width=True)
                    # inserting accept button
                    submit_button = st.form_submit_button(label='Submit')
                    from time import time
                    if submit_button:
                        ts = time()
                        
                        pred_cls = predict_and_gradcam (load_model (), image, device)
                        logger.debug ('classified image created')
                        response_code = 200
                        te = time()
                        te -= ts
                        logger.info('Process time: {}'.format(te))

                    else:
                        response_code = 100

    
    if response_code == 200:
        st.pyplot (pred_cls)
        logger.info ('prediction process complited')
        
        st.success ('prediction done successfully! :thumbsup:')