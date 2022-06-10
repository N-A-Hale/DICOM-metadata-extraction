import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from pydicom import dcmread

### Reading DICOM metadata ###

metadata = dcmread("path/to/file.dcm")

### Extracting tag values (Patient ID here) ###

tag_id = tfio.image.dicom_tags.PatientID
image_bytes = tf.io.read_file("path/to/file.dcm")
tag_value = tfio.image.decode_dicom_data(image_bytes, tag_id).numpy().decode('UTF-8')

### Viewing .dcm image ###

my_image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
plt.imshow(np.squeeze(my_image.numpy()))


### Thanks ###

"""@misc{marcelo_lerendegui_2019_3337331,
    author       = {Marcelo Lerendegui and Ouwen Huang},
    title        = {Tensorflow Dicom Decoder},
    month        = jul,
    year         = 2019,
    doi          = {10.5281/zenodo.3337331},
    url          = {<a href="https://doi.org/10.5281/zenodo.3337331">https://doi.org/10.5281/zenodo.3337331</a>}}"""
