 
import cv2
import imageio
import numpy as np
import skimage.morphology
import torch
from PIL import Image
import torchvision

class SemanticMask():
    def __init__(self, model) -> None:
        self.model = model
 
    def run_maskrcnn(self, img_path):
        threshold = 0.5
        o_image = Image.open(img_path).convert("RGB")
        width, height = o_image.size
        if width > height:
            intHeight = 576
            intWidth = 1024
        else:
            intHeight = 1024
            intWidth = 576

        image = o_image.resize((intWidth, intHeight), Image.Resampling.LANCZOS)

        image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

        tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

        objPredictions = self.model([image_tensor])[0]

        for intMask in range(objPredictions["masks"].size(0)):
            if objPredictions["scores"][intMask].item() > threshold:
                # person, vehicle, accessory, animal, sports
                if objPredictions["labels"][intMask].item() == 1:  # person
                    tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                if (
                    objPredictions["labels"][intMask].item() >= 2
                    and objPredictions["labels"][intMask].item() <= 9
                ):  # vehicle
                    
                    tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                if (
                    objPredictions["labels"][intMask].item() >= 26
                    and objPredictions["labels"][intMask].item() <= 33
                ):  # accessory
                    tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                    
                if (
                    objPredictions["labels"][intMask].item() >= 16
                    and objPredictions["labels"][intMask].item() <= 25
                ):  # animal
                    #tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                    None
                    
                if (
                    objPredictions["labels"][intMask].item() >= 34
                    and objPredictions["labels"][intMask].item() <= 43
                ):  # sports
                    tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                    
                if objPredictions["labels"][intMask].item() == 88:  # teddy bear
                    tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
                    

        npyMask = skimage.morphology.erosion(
            tenHumans.cpu().numpy(), skimage.morphology.disk(1)
        )
        npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
        return npyMask, width, height
    
    def get_mask(self, img_path):
        semantic_mask, W, H = self.run_maskrcnn(img_path)
        semantic_mask = cv2.resize(
            semantic_mask, (W, H), interpolation=cv2.INTER_NEAREST
        )
        return semantic_mask
 