import streamlit as st
from PIL import Image,ImageDraw
import numpy as np
import os
import zipfile
from datetime import datetime as dt
import io

@st.cache
def load_image(img):
    '''
    Opens an Image
    '''
    return Image.open(img)

def square_crop_in_center(img):
    '''
    Crop the image to the biggest centered square
    Takes an image, returns an image
    '''
    width, height = img.size  # Get dimensions
    new_width = min(width, height)
    new_height = min(width, height)

    # New corners of the frame
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    return img.crop((left, top, right, bottom))

def image_to_circle(img):
    '''
    From any image format, provides a circle image (centered)
    Takes a PIL image, returns a PIL image
    '''
    img = img.convert("RGB")
    # Make img as a square if not
    h, w = img.size
    if h != w:
        img = square_crop_in_center(img)

    # Open the input image as numpy array, convert to RGB
    npImage = np.array(img)
    h, w = img.size
    radius = min(h, w)

    # Create same size alpha layer with circle
    alpha = Image.new('L', (h,w),0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, radius, radius], 0, 360, fill=255)

    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)

    # Add alpha layer to RGB
    npImage=np.dstack((npImage,npAlpha))

    # Save with alpha
    return Image.fromarray(npImage)


def image_to_vignette(img, overlay):
    '''
    Adding the CEC layer around the
    Takes an image and the overlay, returns an image
    '''
    img = image_to_circle(img)

    img = img.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Coef used to oversize the overlay, in order to keep image picture quality hihg enough on result. If coaf = 1, output is (300,300) pixels
    coef = 3

    # Making overlay squared (current overlay is 300*302)
    o_w, o_h = overlay.size
    overlay = overlay.resize((o_w * coef, o_h * coef),
                             Image.Resampling.LANCZOS)

    #New overlay size
    o_w, o_h = overlay.size

    # Downsizing the image
    new_size = (220 * coef, 220 * coef)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    i_w, i_h = img_resized.size

    px_vertical_offset = 1 * coef  # it seems transparent circle is not perfectly centered
    offset_to_center = ((o_w - i_w) // 2,
                        (o_h - i_h) // 2 + px_vertical_offset)

    img_resized_tbg = Image.new('RGBA', (o_w, o_h), (255, 255, 255, 0))
    img_resized_tbg.paste(img_resized, offset_to_center)


    return Image.alpha_composite(overlay, img_resized_tbg)

def main():
    # Set valid format for upload, then upload
    valid_images = [".jpg",".jpeg", ".png"]
    files = st.file_uploader(
        label=
        'Sélectionner une ou plusieurs photos (jpg, jpeg ou png)',
        type=valid_images,
        accept_multiple_files=True)

    # Loading overlay
    overlay = load_image("masque.png")

    #Create compressed zip archive and add files
    zip_name = '_'.join([dt.today().strftime('%Y-%m-%d_%H:%M:%S'),
                        'vignettes_cec'])

    with zipfile.ZipFile(zip_name, mode='w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            if file is not None:
                file_name_circle = f"{os.path.splitext(file.name)[0]}_cercle{os.path.splitext(file.name)[1]}"
                file_name_vignette = f"{os.path.splitext(file.name)[0]}_vignette{os.path.splitext(file.name)[1]}"

                # Creating the circle image
                img = load_image(file)
                img_circle = image_to_circle(img)

                img_circle_buffer = io.BytesIO()
                img_circle.save(img_circle_buffer, format="PNG")
                z.writestr(zinfo_or_arcname=file_name_circle, data=img_circle_buffer.getvalue())

                # Creating the vignette image
                img_vignette_buffer = io.BytesIO()
                img_vignette = image_to_vignette(img, overlay)

                img_vignette.save(img_vignette_buffer, format="PNG")
                z.writestr(zinfo_or_arcname=file_name_vignette, data=img_vignette_buffer.getvalue())

                img_circle_buffer.close()
                img_vignette_buffer.close()

                #Display images on website
                cols = st.columns(2)
                cols[0].image(img_circle.resize((50,50)))
                cols[1].image(img_vignette.resize((50, 50)))

    # Download button appears if some images have been processed
    if len(files) > 0:
        with open(zip_name, mode='rb') as z:
            st.download_button(
                        label="Télécharger les vignettes",
                        data=z,
                        file_name=zip_name,
                        mime="application/zip")

    #Remove the temp directory created in the repo
    os.remove(zip_name)


########################################################
########################################################
# WEBSITE STARTS HERE
########################################################
########################################################

title_cols = st.columns([2, 6, 2])

title_cols[1].image('LogoCEC.png')

st.title('Editeur de vignette')

st.markdown('''
            Ce site permet, à partir d'une photo de profil, d'obtenir:
            ''')
st.markdown(" - sa version rognée en rond ")
st.markdown(" - ainsi que la vignette associée  ")
st.markdown('''
            Il est possible d'éditer plusieurs photos à la fois.
            ''')
st.markdown('')

main()

st.markdown('')
st.caption('''
Version Beta, pour usage interne à la CEC. Réalisé par Arthur C de la CEC BL&A (contact en cas de problème).
''')
