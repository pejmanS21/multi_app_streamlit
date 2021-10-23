import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from PIL import Image
from skimage import measure
from skimage.transform import resize

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, concatenate, add

import streamlit as st

# """-----------------------------------logger file------------------------------------------"""
import logging

logger = logging.getLogger ('detection_app')
#adding level to logger
logger.setLevel (logging.DEBUG)

formatter = logging.Formatter ('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler ('../logs/multi_app.log')
# adding formatter to file_handler
file_handler.setFormatter (formatter)

# adding file & stream handler to logger
logger.addHandler (file_handler)


def app ():

    def preprocess (file):
        """ reading file from disk and change it to numpy array """
        # read image in grayscale mode
        image = np.array (Image.open (file).convert ('L'))
        # log the image shape
        logger.info ('Shape of input image:{}'.format (image.shape))
        # resize both image
        image = resize (image, (128, 128), mode = 'symmetric')
        # add trailing channel dimension
        image = np.expand_dims (image, -1).astype (np.float32)
        # add batch size of 1
        image = np.expand_dims (image, 0)

        # log the complitation of preprocess
        logger.debug ('Preprocess done')

        return image


    def create_downsample (channels, inputs):
        """ bn + conv + leaky relu + maxpool """
        x = BatchNormalization (momentum = 0.9)(inputs)
        x = LeakyReLU (0)(x)
        x = Conv2D (channels, 1, padding = 'same', use_bias = False)(x)
        x = MaxPool2D (2)(x)

        return x


    def create_resblock (channels, inputs):
        """ a convelution block with residual connection """
        x = BatchNormalization (momentum = 0.9)(inputs)
        x = LeakyReLU (0)(x)
        x = Conv2D (channels, 3, padding='same', use_bias = False)(x)
        x = BatchNormalization (momentum = 0.9)(x)
        x = LeakyReLU (0)(x)
        x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)

        #Added Start
        x = BatchNormalization (momentum = 0.9)(x)
        x = LeakyReLU (0)(x)
        x = Conv2D (channels, 3, padding = 'same', use_bias = False)(x)
        #Added End
        
        addInput = x;
        resBlockOut = add ([addInput, inputs])
        out = concatenate([resBlockOut, addInput], axis = 3)
        out = Conv2D (channels, 1, padding = 'same', use_bias = False)(out)
        return out


    def create_network (weight_path, input_size = 128, channels = 16, n_blocks = 2, depth = 3):
        """ create final network like unet architecture """
        # input
        inputs = Input (shape = (input_size, input_size, 1))
        x = Conv2D (channels, 3, padding = 'same', use_bias = False)(inputs)
        # residual blocks
        for d in range (depth):
            channels = channels * 2
            x = create_downsample (channels, x)
            for b in range (n_blocks):
                x = create_resblock (channels, x)
        # output
        x = BatchNormalization (momentum = 0.9)(x)
        x = LeakyReLU (0)(x)
        x = Conv2D (1, 1, activation = 'sigmoid')(x)
        outputs = UpSampling2D (2**depth)(x)
        model = Model (inputs = inputs, outputs = outputs)
        logger.debug ('model created!')
        model.load_weights (weight_path)
        logger.debug ('weights loaded')
        
        return model

    def show_preds_with_bboxes (model, image):
        """ showing selected picture with bounding box """
        # predict batch of images
        pred = model.predict (image)
        logger.debug ('prediction done!')
        # create figure
        fig, ax = plt.subplots ()
        # plot image
        ax.imshow (image [0][:, :, 0])
        ax.axis ('off')
        # threshold predicted mask
        comp = pred [0][:, :, 0] > 0.5
        comp = comp.astype (np.float32)
        # apply connected components
        comp = measure.label (comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops (comp):
            if region:
                predictionString = 'Pneumonia Detected!'
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                ax.add_patch (patches.Rectangle((x,y), width,height,linewidth = 2, edgecolor = 'b',facecolor = 'none'))
            else:
                predictionString = 'No Pneumonia Detected'
        
        ax.set_title (predictionString)
        plt.show()
        logger.debug ('fig was shown!')
        return fig

    @st.cache(allow_output_mutation=True)
    def load_model ():

        # Detection
        return create_network (weight_path = '../weights/RSNA_OD.h5')

    response_code = 100
    
    st.markdown("""
                # Detection 
                A Pneumonia Detection app that use `Tensorflow` as framework, based on **RSNA** Dataset.
                Upload your CXR Image and hit the `Submit` button to get predictions.

                -------------------------------------------------------------------------    
    """)

    with st.form(key='Detection'):
        with st.sidebar:
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                
                if file is not None:
                    image = preprocess (file)

                    st.image(image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')
                    from time import time
                    if submit_button:
                        ts = time()

                        pred_det = show_preds_with_bboxes (load_model (), image)
                        logger.debug ('detection image created')
                        # rcm = resourceManagement ()
                        response_code = 200
                        te = time()
                        te -= ts
                        logger.info('Process time: {}'.format(te))

                    else:
                        response_code = 100

    
    if response_code == 200:
        st.pyplot (pred_det)
        logger.info ('detection process complited')
        
        st.success ('Detection Process Complited! :thumbsup:')