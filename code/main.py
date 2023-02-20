from train import train_VIT
import os
import argparse

import albumentations as A
import albumentations.pytorch

from preprocessing import preprocess_image
import pickle
import torch
import torchvision


def main():
    default_parser = argparse.ArgumentParser()

    default_parser.add_argument('-t', '--train', action='store_true')
    default_parser.add_argument('-p', '--predict', action='store')

    args = default_parser.parse_args()

    if args.train:
        train_VIT()
        return
    
    if args.predict:

        models_list = os.listdir('/model')

        # Check trained models
        if len(models_list) <= 2:
            print('Trained models not found! Call program with -t flag first!')
            return 

        # Load trained model
        with open(os.path.join('/model', sorted(models_list)[-1]), 'rb') as file:
            model = pickle.load(file)
            model.eval()

        image_path = os.path.join('/inference', args.predict)

        if os.path.exists(image_path) and os.path.isfile(image_path):
            print('Image preprocessing started!')
            preprocessed_image = preprocess_image(image_path)
            print('Image preprocessing finished!')

            img_transforms = A.Compose(
                [
                    A.augmentations.geometric.resize.Resize(384, 384),
                    A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    albumentations.pytorch.ToTensorV2(),
                ]
            )

            preprocessed_image = img_transforms(image=preprocessed_image)['image']
            preprocessed_image = torch.unsqueeze(torch.FloatTensor(preprocessed_image), dim=0)

            result = model(preprocessed_image)[0]

            print('Predicted label: ', torch.argmax(result).item())

        else:
            print('Error image path!')
            return


if __name__ == '__main__':
    main()