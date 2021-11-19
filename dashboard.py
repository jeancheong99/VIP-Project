import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')


def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image

# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

# def resize(image, height, width):
#   image = tf.image.resize(image, [height, width],
#               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#   return image

# def random_crop(input_image, real_image):
#   stacked_image = tf.stack([input_image, real_image], axis=0)
#   cropped_image = tf.image.random_crop(
#       stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

#   return cropped_image[0], cropped_image[1]

def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image

# @tf.function()
# def random_jitter(input_image, real_image):
#   # Resizing to 286x286
#   input_image, real_image = resize(input_image, real_image, 286, 286)

#   # Random cropping back to 256x256
#   input_image, real_image = random_crop(input_image, real_image)

#   if tf.random.uniform(()) > 0.5:
#     # Random mirroring
#     input_image = tf.image.flip_left_right(input_image)
#     real_image = tf.image.flip_left_right(real_image)

#   return input_image, real_image

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Restore
def restore_checkpoint(ckpt_dir):
  checkpoint_dir = ckpt_dir
  print('CHECKPOINT FILE => ', tf.train.latest_checkpoint(checkpoint_dir))
  print('CHECKPOINT LOAD STATUS => ',checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial())

#############################################################################################################################################
################################# D A S H B O A R D #########################################################################################
### COLORS USED #############################################################################################################################
building_wall = '#002fff' # blue 
building2_wall = '#0000de' # medium blue
window = '#007ffd' # light blue / dodgerblue

wall2 = '#fea000' # orange
wall3 = '#fd4f00' # orange red
color5 = '#fb5000' # orange red
pillars = '#fb0000' # red
basement_shoplot = '#ae0000' # dark red

balcony = '#bdff3b' # green yellow
#############################################################################################################################################
def hex2rgb(h):
    h = h.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    a = np.asarray( a, dtype='float32' ) / 255.0
    R, G, B = background
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def load_model(model):
  if model == 'Shoe':
    ckpt_dir = './checkpoints/shoe/'
    restore_checkpoint(ckpt_dir)

  elif model == 'Handbag':
    ckpt_dir = './checkpoints/handbag/'
    restore_checkpoint(ckpt_dir)

  elif model == 'Coloriser':
    ckpt_dir = './checkpoints/coloriser/'
    restore_checkpoint(ckpt_dir)

  else:
    ckpt_dir = './checkpoints/facade/'
    restore_checkpoint(ckpt_dir)

def predict(generator, drew):
    drew = drew.copy()
    drew = rgba2rgb(drew)
    drew = resize(drew, (256, 256))
    drew = tf.expand_dims(drew, axis=0) 
    #print(drew)
    fig, ax = plt.subplots(figsize=(6,6))
    prediction = generator(drew, training=True)
    ax.imshow(prediction[0]*0.5+0.5)
    plt.axis('off')
    st.pyplot(fig)

#############################################################################################################################################
st.title('Smart Sketch: AI Art Application turns Simple Sketches into Photorealistic Images')

model = st.sidebar.selectbox("Select a model: ", ('Shoe', 'Handbag', 'Coloriser', 'Facade'))

if model == 'Facade':
  drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "rect", "transform"))

  if drawing_mode == 'freedraw':  
      stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
      stroke_color = st.sidebar.color_picker("Stroke color hex: ")

  fill_color_ = st.sidebar.color_picker("Fill color hex: ")
  bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
  bg_image = st.sidebar.file_uploader("Upload image:", type=["png", "jpg"])

  if drawing_mode != 'freedraw':  
      stroke_width = None
      stroke_color = fill_color_

  column1, column2 = st.columns(2)
  # Create a canvas component
  with column1:
    if bg_image is None:
      canvas_result = st_canvas(
          stroke_width=stroke_width,
          stroke_color=stroke_color,
          fill_color=fill_color_,
          background_color=bg_color,
          background_image=Image.open(bg_image).convert('RGB') if bg_image else None,
          update_streamlit=False,
          display_toolbar=True,
          drawing_mode=drawing_mode,
          key="canvas",
          height=400,
          width=400
      )
    else:
      bg_image2 = Image.open(bg_image)#.convert('RGB')
      plt.figure(figsize=(10,10))
      plt.imshow(bg_image2)
      plt.axis('off')
      st.pyplot()

  if st.button('Generate Image'):
    with column2:
      if bg_image != None:
        drawed_image = Image.open(bg_image)
        drawed_image = np.array(drawed_image)
        print('NDIM (uploaded): ', drawed_image.ndim)
      else:
        drawed_image = canvas_result.image_data
        drawed_image = rgba2rgb(drawed_image)
        #plt.figure(figsize=(10,10))
        # plt.imshow(drawed_image)
        # plt.axis('off')
        # st.pyplot()
      drawed_image = resize(drawed_image, (256, 256))
      #drawed_image = normalize(drawed_image)
      
      st.write('\n')
      st.subheader("Input image after resized:")
      plt.imshow(drawed_image)
      plt.axis('off')
      st.pyplot()
      
      drawed_image = tf.expand_dims(drawed_image, axis=0)
      print(drawed_image.shape)

      fig, ax = plt.subplots(figsize=(10, 10))
      plt.axis('off')
      prediction = generator(drawed_image, training=True)
      st.subheader("Output from model:")
      im = ax.imshow(prediction[0]*0.5+0.5)
      #plt.title('Output Image from model:')
      st.pyplot()

else:
  stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
  stroke_color = st.sidebar.color_picker("Stroke color hex: ")
  drawing_mode = st.sidebar.selectbox(
      "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
  )

  row1_c1, row1_c2 = st.columns(2)

  with row1_c1:
      canvas_result = st_canvas(
          fill_color="rgb(255, 165, 0)",  # Fixed fill color with some opacity
          stroke_width=stroke_width,
          stroke_color=stroke_color,
          height=400,
          width=400,
          drawing_mode=drawing_mode,
          key='canvas',
      )

  load_model(model)

  if st.button('Generate Image'):
      with row1_c2:
          predict(generator=generator, drew=canvas_result.image_data)
