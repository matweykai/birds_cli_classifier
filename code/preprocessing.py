import cv2
import os
import numpy as np
from tqdm import tqdm
from chainercv.links import YOLOv3
from chainercv.utils import read_image


def preprocess_images_folder(img_dir_path: str, crop_model) -> None:
    # Load images
    file_name_list = sorted(os.listdir(img_dir_path))
    raw_imgs_list = [read_image(os.path.join(img_dir_path, temp_img)) for temp_img in file_name_list]
    
    # Predict bounding boxes
    model_predictions = crop_model.predict(raw_imgs_list)
    
    #print(model_predictions)
    
    for temp_raw_img, temp_bbox, temp_old_file_name in zip(raw_imgs_list, model_predictions[0], file_name_list):
        orig_img = np.swapaxes(np.swapaxes(np.array(temp_raw_img, dtype=np.uint8), 0, 1), 1, 2)

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        if temp_bbox.size > 0:
            y0, x0, y1, x1 = temp_bbox[0]

            cropped_img = orig_img[int(y0)-10:int(y1)+10, int(x0)-10:int(x1)+10, :]

            new_folder_path = os.path.join(img_dir_path.replace('input', 'modified'))
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            if cropped_img.size > 0:
                cv2.imwrite(os.path.join(new_folder_path, temp_old_file_name), cropped_img)
                continue
        
        cv2.imwrite(os.path.join(new_folder_path, temp_old_file_name), orig_img)


def main():
    crop_model = YOLOv3(pretrained_model='voc0712')

    train_dir = os.getenv('IMAGES_DIRECTORY')

    counter = 1

    for temp_folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, temp_folder)

        if os.path.isdir(folder_path):
            preprocess_images_folder(folder_path, crop_model)

        print(f"Processed #{counter}")
        counter += 1 



if __name__ == '__main__':
    main()