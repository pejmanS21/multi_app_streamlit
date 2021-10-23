import streamlit as st
import segmentation_app
import detection_app
import classification_app
from time import time
from resource_manager import *


PAGES = {
    "None": None,
    "Lung Segmentation": segmentation_app,
    "Detection": detection_app,
    "Classification": classification_app
}

st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))

page = PAGES[selection]
if page is not None:
    page.app()



