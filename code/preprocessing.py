import cv2
import os
import numpy as np
from chainercv.links import YOLOv3
from chainercv.utils import read_image
import pickle

# Load crop model from binary file
try:
    with open('/model/crop_model.bin', 'rb') as file:
        crop_model = pickle.load(file)
except Exception:
    print('Problems with crop model!')
    exit()


def preprocess_image(img_path: str) -> np.array:
    img = read_image(img_path)

    model_prediction = crop_model.predict([img])

    # Change image format from (C, H, W) to (H, W, C)
    orig_img = np.swapaxes(np.swapaxes(np.array(img, dtype=np.uint8), 0, 1), 1, 2)

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    pred_bbox = model_prediction[0][0]

    if pred_bbox.size > 0:
        y0, x0, y1, x1 = pred_bbox[0]

        cropped_img = orig_img[int(y0)-10:int(y1)+10, int(x0)-10:int(x1)+10, :]

        if cropped_img.size > 0:
            return cropped_img

    return orig_img


def preprocess_images_folder(img_dir_path: str) -> None:
    # Load images
    file_name_list = sorted(os.listdir(img_dir_path))
    raw_imgs_list = [read_image(os.path.join(img_dir_path, temp_img)) for temp_img in file_name_list]
    
    # Predict bounding boxes
    model_predictions = crop_model.predict(raw_imgs_list)
    
    for temp_raw_img, temp_bbox, temp_old_file_name in zip(raw_imgs_list, model_predictions[0], file_name_list):
        # Change image format from (C, H, W) to (H, W, C)
        orig_img = np.swapaxes(np.swapaxes(np.array(temp_raw_img, dtype=np.uint8), 0, 1), 1, 2)

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        # Create preprocessing directories
        new_folder_path = os.path.join(img_dir_path.replace('raw', 'preprocessed'))
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        if temp_bbox.size > 0:
            y0, x0, y1, x1 = temp_bbox[0]

            cropped_img = orig_img[int(y0)-10:int(y1)+10, int(x0)-10:int(x1)+10, :]

            if cropped_img.size > 0:
                cv2.imwrite(os.path.join(new_folder_path, temp_old_file_name), cropped_img)
                continue
        
        cv2.imwrite(os.path.join(new_folder_path, temp_old_file_name), orig_img)
