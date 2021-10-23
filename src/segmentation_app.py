import streamlit as st
import numpy as np
import os
import cv2
from time import time
from resource_manager import *
from PIL import Image, ImageOps
import logging
from typing import Tuple
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from base_classes import SegmentationModel

# """-----------------------------------logger file------------------------------------------"""
logger = logging.getLogger("Streamlit_Server")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('../logs/multi_app.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def app():
    #"""-----------------------------------Residual U-Net------------------------------------"""
    # """
    #     :summary:  Residual U-Net implementation
    #     :Framework: tensorflow
    #     :Libraries: 

    # """
    # """DICE"""
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    # """IoU(Jaccard)"""
    def jaccard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


    def jaccard_coef_loss(y_true, y_pred):
        return 1 - jaccard_coef(y_true, y_pred)

    # """
    #     :Custom layers for Res U-Net:
    #     :BatchNormalization:
    # """
    def bn_act(x, act=True):
        x = tensorflow.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tensorflow.keras.layers.Activation("relu")(x)
        return x

    # """
    #     :Custom layers for Res U-Net:    
    # """
    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = bn_act(x)
        conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    # """
    #     :Custom layers for Res U-Net:  
    # """
    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = tensorflow.keras.layers.Add()([conv, shortcut])
        return output

    # """
    #     :Custom layers for Res U-Net:
    #     :Residual layer:   
    # """
    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = tensorflow.keras.layers.Add()([shortcut, res])
        return output

    # """
    #     :Custom layers for Res U-Net:
    #     :UpSample layer:   
    # """
    def upsample_concat_block(x, xskip):
        u = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
        c = tensorflow.keras.layers.Concatenate()([u, xskip])
        return c


    # """
    #     :Summary:  Residual U-Net implementation
    #     :param pretrained_weights: where the pretarined_weight stored
    #     :param input_size: input shape for the model to be built with
    #     :Description: get a CXR image with size of (256, 256, 1) and return a mask with same size
    #     :Result: 99% AUC, 99% Accuracy
    # """
    @st.cache(allow_output_mutation=True)
    def ResUnet_Builder(pretrained_weights: str, pretrained: bool = True,
                    input_size: Tuple[int, int, int] = (256, 256, 1)):
        f = [16, 32, 64, 128, 256]
        inputs = tensorflow.keras.layers.Input(input_size)

        # Encoder
        e0 = inputs
        e1 = stem(e0, f[0])
        e2 = residual_block(e1, f[1], strides=2)
        e3 = residual_block(e2, f[2], strides=2)
        e4 = residual_block(e3, f[3], strides=2)
        e5 = residual_block(e4, f[4], strides=2)

        # Bridge
        b0 = conv_block(e5, f[4], strides=1)
        b1 = conv_block(b0, f[4], strides=1)

        # Decoder
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block(u1, f[4])

        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block(u2, f[3])

        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block(u3, f[2])

        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block(u4, f[1])

        outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
        model = tensorflow.keras.models.Model(inputs, outputs)

        metrics = [dice_coef, jaccard_coef,
                'binary_accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
        
        loss = [dice_coef_loss,
                jaccard_coef_loss,
                'binary_crossentropy']

        adam = tensorflow.keras.optimizers.Adam()
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

        if pretrained == True:
            # """----- Load Weight -----"""
            model.load_weights(pretrained_weights)

        return model

    # """---------------------------------data pre_process------------------------------------"""
    # """
    #     :summary:  functions for load data from streamlit
    #     :dtype:  PIL to Numpy array and CV2 format
    #     :Normalization:  [0, 255] -> [-1, 1]
    #     :Shape: (1, 256, 256, 1)

    # """
    reference_shape = (256, 256, 1)

    def load_data(image, pre_process="Original", dim=256):
        """
            load images for models when you run streamlit.
            :image: uploaded image in streamlit.
            :dim: shape of image 
            :pre_process: pre-process for loaded images (DHE or Original)
        """
        image = ImageOps.fit(image, (dim, dim))
        image = np.asarray(image)
        # pil to cv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check channel
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compatible size
            if image.shape != reference_shape:
                image = cv2.resize(image, (dim, dim))

        # selected pre_process
        if pre_process == "DHE":
            image = cv2.equalizeHist(image)

        # reshape & normalize
        image = image.reshape(1, dim, dim, 1)
        image = (image - 127.0) / 127.0
        logger.info("Process Complete!")
        return image

    def stream_data(file, pre_process="Original", dim=256):
        """
            :file:  path that pointed to the uploaded image in streamlit
            :pre_process:   Original or DHE for image
            :dim:  size for image
        """
        image = Image.open(file)
        image = ImageOps.fit(image, (dim, dim))
        image = np.asarray(image)
        # pil to cv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check channel
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compatible size
            if image.shape != reference_shape:
                image = cv2.resize(image, (dim, dim))

        # selected pre_process
        if pre_process == "DHE":
            image = cv2.equalizeHist(image)

        # reshape & normalize
        image = image.reshape(1, dim, dim, 1)
        image = image / 255.
        return image

    # """------------------------------Save image & mask for visualization-------------------------"""
    # """
    #     :summary:  save predicted mask in send it to frontend
    #     :param X: input CXR image
    #     :param predicted: lung mask    
    #     :output: attach CXR image and mask together.
    # """
    def visualize_output(X, predicted, mask_type: str = 'White Mask'):
         
        # X.shape = (n, 256, 256, 1)
        output_figure = np.zeros((X.shape[1] * X.shape[0], X.shape[2] * 2, 1))

        for i in range(len(X)):
            output_figure[i * 256: (i + 1) * 256,
            0: 1 * 256] = ((X[i] * 127.) + 127.)
            
            if mask_type == 'White Mask':
                output_figure[i * 256: (i + 1) * 256,
                1 * 256: 2 * 256] = (predicted[i] * 255)
            else:
                output_figure[i * 256: (i + 1) * 256,
                1 * 256: 2 * 256] = ((predicted[i] * 127.) + 127.)

        fig_shape = np.shape(output_figure)
        output_figure = output_figure.reshape((fig_shape[0], fig_shape[1]))
        cv2.imwrite('../images/output_figure.png', output_figure)
        logger.info('Output figure saved!')
        return output_figure

    # """---------------------------------Lung Segmentor------------------------------------"""
    def segmented_creation(image, pred):
        """
            Description: get CXR Images and return lungs area as numpy array.
            
            Params:{
                    images: CXR Images with shape (n_images, 256, 256, 1).
                    mask: predicted white mask.   (n_images, 256, 256, 1)
                    }

            Outputs: only lungs in CXR images with shape (n_images, 256, 256)

        """
        pred = pred.reshape(len(pred), 256, 256)

        X_gray = image[:, :, :, 0]
        segmented_image = np.zeros((image.shape[0],
                                    image.shape[1],
                                    image.shape[2]))
        ## create lung area
        for i in range(len(X_gray)):
            lung_mask = pred[i].copy()
            lung_mask = lung_mask * 255.
            im = X_gray[i].copy()
            im[lung_mask < 127] = 0
            segmented_image[i] = im

        return segmented_image

    # """----------------------------------Streamlit APP---------------------------------------------"""
    st.markdown('''
                # Lung Segmentation App
                A `U-Net` base model with `Residual` blocks.**Accuracy**: `99%`.

                Select your **CXR** image and **pre_process** then hit the `Submit` button to get **Detected mask**.           

                ------------------------------------------------------
                ''')
    response_code = 400
    model = ResUnet_Builder(pretrained_weights="../weights/cxr_seg_res_unet.hdf5", 
                            pretrained=True, 
                            input_size=(256, 256, 1))   

    with st.form(key='segmentation'):
        with st.sidebar:
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                if file is not None:
                    processed_image = stream_data(file, pre_process=pre_process)
                    image = load_data(Image.open(file), pre_process)
                    image = image.reshape(1, 256, 256, 1)

                    mask_type = st.sidebar.radio("Mask Type", ["White Mask", "Lung Mask"])
                    st.image(processed_image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')
                    from time import time
                    if submit_button:
                        ts = time()
                        mask = model.predict(image)
                        resources = Resource_Manager()
                        resources = resources.monitor()
                        te = time()
                        te -= ts

                        if mask_type == "Lung Mask":
                            mask = segmented_creation(image, mask)
                            mask = mask.reshape(1, 256, 256, 1)
                            
                        output_figure = visualize_output(image, mask, mask_type)
                        
                        logger.info(':PreProcess: {}:Process time: {}'.format(pre_process, te))
                        response_code = 200
                    else: response_code = 400

    
    if response_code == 200:
        st.write("### CXR Image & Detected mask")
        st.image(output_figure / 255., use_column_width=False)
        
        st.success('Mask Detected successfully! :thumbsup:')