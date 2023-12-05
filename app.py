import streamlit as st
from PIL import Image,ImageDraw
import numpy as np
import os
import zipfile
from datetime import datetime as dt
import io

@st.cache_data
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

    # Coef used to oversize the overlay, in order to keep image picture quality hihg enough on result. If coef = 1, output is (300,300) pixels
    coef = 2

    # Making overlay squared (current overlay is 300*302)
    o_w, o_h = overlay.size
    overlay = overlay.resize((o_w * coef, o_h * coef),
                             Image.Resampling.LANCZOS)

    #New overlay size
    o_w, o_h = overlay.size

    # Downsizing the image
    inner_circle_size = 377
    new_size = (inner_circle_size * coef, inner_circle_size * coef)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    i_w, i_h = img_resized.size

    px_vertical_offset = 0 * coef  # if transparent circle is not perfectly vertically centered, O px offset in this version
    offset_to_center = ((o_w - i_w) // 2,
                        (o_h - i_h) // 2 + px_vertical_offset)

    img_resized_tbg = Image.new('RGBA', (o_w, o_h), (255, 255, 255, 0))
    img_resized_tbg.paste(img_resized, offset_to_center)


    return Image.alpha_composite(overlay, img_resized_tbg)

def main(cercle,vignette):
    # Set valid format for upload, then upload
    valid_images = [".jpg",".jpeg", ".png"]
    files = st.file_uploader(
        label=
        'Sélectionner une ou plusieurs photos (jpg, jpeg ou png)',
        type=valid_images,
        accept_multiple_files=True)

    # Loading overlay
    overlay = load_image("LK_Profil_CEC_600x600_vide.png")

    #Create compressed zip archive and add files
    zip_name = '_'.join([dt.today().strftime('%Y-%m-%d_%H:%M:%S'),
                        'vignettes_cec'])

    with zipfile.ZipFile(zip_name, mode='w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            if file is not None:

                img = load_image(file)

                # Creating the circle image
                if cercle:
                    file_name_circle = f"{os.path.splitext(file.name)[0]}_cercle{os.path.splitext(file.name)[1]}"
                    img_circle = image_to_circle(img)
                    img_circle_buffer = io.BytesIO()
                    img_circle.save(img_circle_buffer, format="PNG")
                    z.writestr(zinfo_or_arcname=file_name_circle, data=img_circle_buffer.getvalue())
                    #img_circle_buffer.close()

                # Creating the vignette image
                if vignette:
                    file_name_vignette = f"{os.path.splitext(file.name)[0]}_vignette{os.path.splitext(file.name)[1]}"
                    img_vignette_buffer = io.BytesIO()
                    img_vignette = image_to_vignette(img, overlay)
                    img_vignette.save(img_vignette_buffer, format="PNG")
                    z.writestr(zinfo_or_arcname=file_name_vignette, data=img_vignette_buffer.getvalue())
                    #img_vignette_buffer.close()

                #Display images on website
                if vignette and cercle:
                    cols = st.columns(2)
                    cols[0].image(img_circle.resize((50,50)))
                    cols[1].image(img_vignette.resize((50, 50)))
                elif vignette and not cercle:
                    st.image(img_vignette.resize((50, 50)))
                elif not vignette and cercle:
                    st.image(img_circle.resize((50,50)))
                else:
                    continue

    # Download button appears if some images have been processed

    if len(files) > 0:
        if (vignette and cercle) or len(files) > 1:
            with open(zip_name, mode='rb') as z:
                st.download_button(
                            label="Télécharger",
                            data=z,
                            file_name=zip_name,
                            mime="application/zip")
        elif (vignette and not cercle) and len(files) == 1:
            st.download_button(label="Télécharger",
                               data=img_vignette_buffer,
                               file_name=file_name_vignette,
                               mime="image/png")
        elif (not vignette and cercle) and len(files) == 1:
            st.download_button(label="Télécharger",
                               data=img_circle_buffer,
                               file_name=file_name_circle,
                               mime="image/png")
    #Remove the temp directory created in the repo
    os.remove(zip_name)
    try:
        img_circle_buffer.close()
        img_vignette_buffer.close()
    except Exception:
        pass

########################################################
########################################################
# WEBSITE STARTS HERE
########################################################
########################################################

title_cols = st.columns([2, 6, 2])

title_cols[1].image('LogoCEC.png')

st.title('Editeur de vignette')

st.markdown('''
            A partir d'une photo de profil, ce site permet d'obtenir:
            ''')
img_cercle_controle = load_image("cercle_controle.png")
img_vignette_controle = load_image("vignette_controle.png")

c_bullet = st.columns(2)
cercle = c_bullet[0].checkbox('sa version rognée en rond',
                              value=False)
c_bullet[1].image(img_cercle_controle.resize((50, 50)))

v_bullet = st.columns(2)
vignette = v_bullet[0].checkbox(
    'ainsi que la vignette CEC associée', value=True,
    disabled=False)  # Set disable=True to have the checkbox mandatory
v_bullet[1].image(img_vignette_controle.resize((50, 50)))

st.markdown('''
            Il est possible d'éditer plusieurs photos à la fois.
            ''')
st.markdown('')



main(cercle,vignette)

st.markdown('')
st.caption('''
Version Beta, pour usage interne à la CEC. Réalisé par Arthur C de la CEC BL&A (contact en cas de problème).
''')
