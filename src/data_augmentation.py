import sys
import os
import torchvision.transforms as transforms
from PIL import Image
from exception import CustomException
from logger import logging
import argparse

# Define the data augmentation transforms
data_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]),
])

def augment_images(directory, num_augmented_samples=3):
    try:
    
        # List all image files in the original directory
        image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for i, file_name in enumerate(image_files):
            # Load the image
            image_path = os.path.join(directory, file_name)
            image = Image.open(image_path)
            
            # Perform data augmentation on the image
            for j in range(num_augmented_samples): # Generate 3 augmentated samples for each image
                augment_images = data_transforms(image)
                augment_images_path = os.path.join(directory, f"augmented_{i}_{j}.jpg")
                
                # Convert the tensor back to PIL image and save
                Image.fromarray((augment_images*255).permute(1, 2, 0).numpy().astype('uint8')).save(augment_images_path)
                logging.info('Data Augmentation completed!')
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == '__main__':
    print("Augmenting")
    parser = argparse.ArgumentParser(description='Data Augmentation')
    parser.add_argument('-num_augmented_samples', type=int, default=3, help="Number of augmented sampled for each image")
    args = parser.parse_args()
    augment_images("D:\\hand-safety-detection\\images\\augmented", args.num_augmented_samples)
    print("Completed")   

