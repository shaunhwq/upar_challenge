import argparse
import os
from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

from models.backbone import swin_transformer2
from models.base_block import FeatClassifier
from models.model_factory import build_classifier, build_backbone
from tools.function import get_reload_weight


class C2TNetHelper:
    def __init__(
        self,
        weights_path: str="checkpoints/best_model.pth",
        input_height: int=256,
        input_width: int=128,
        device: str="cpu"
    ):
        self.device = device
        self.validation_transform = T.Compose([
            T.Resize((input_height, input_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Build model
        backbone, c_output = build_backbone("swin_b", multi_scale=False)
        classifier = build_classifier("linear")(
            nattr=40,
            c_in=2048,
            bn=False,
            pool="avg",
            scale=1
        )
        self.model = FeatClassifier(backbone, classifier)
        load_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(load_dict, OrderedDict):
            pretrain_dict = load_dict
        else:
            pretrain_dict = load_dict['state_dicts']
        pretrain_dict = {key.replace('module.', ''): value for key, value in pretrain_dict.items()}
        self.model.load_state_dict(pretrain_dict, strict=True)
        self.model.to(device)

    def post_process(self, model_output) -> np.array:
        valid_logits, attns = model_output
        valid_probs = torch.sigmoid(valid_logits[0])

        valid_prob = valid_probs[0].cpu().numpy().tolist()

        # Get Textual Description
        textual_description = self._get_textual_description(valid_prob)

        return textual_description, valid_prob

    @staticmethod
    def _get_textual_description(probabilities: List[float]) -> dict:
        """
        :param probabilities: A list of probabilities corresponding to different textual descriptions of the person in the image
        :returns: Dictionary with the description of person
        """
        textual_description = dict(
            age = ["Young", "Adult", "Old"][probabilities[:3].index(max(probabilities[:3]))],
            gender = "Female" if probabilities[3] > 0.5 else "Male",
            hair_length = ["Short", "Long", "Bald"][probabilities[4: 7].index(max(probabilities[4: 7]))],
            upper_body_length = "Short" if probabilities[7] > 0.5 else "Long",
            upper_body_color = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"][probabilities[8: 20].index(max(probabilities[8: 20]))],
            lower_body_length = "Short" if probabilities[20] > 0.5 else "Long",
            lower_body_color = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"][probabilities[21: 33].index(max(probabilities[21: 33]))],
            lower_body_type = ["Trousers & Shorts", "Skirt & Dress"][probabilities[33: 35].index(max(probabilities[33: 35]))],
            backpack = True if probabilities[35] > 0.5 else False,
            bag = True if probabilities[36] > 0.5 else False,
        )
        normal_glasses_prob = probabilities[37]
        sun_glasses_prob = probabilities[38]
        if normal_glasses_prob < 0.5 and sun_glasses_prob < 0.5:
            glasses = "None"
        else:
            glasses = ["Normal", "Sun"][0 if normal_glasses_prob > sun_glasses_prob else 1]

        textual_description["glasses"] = glasses
        textual_description["hat"] = True if probabilities[39] > 0.5 else False
        return textual_description

    def pre_process(self, image: np.array):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        input_tensor = self.validation_transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        return input_tensor

    @torch.no_grad()
    def infer_single(self, image: np.array):
        input_tensor = self.pre_process(image)
        model_output = self.model(input_tensor)
        output = self.post_process(model_output)
        return output


if __name__ == "__main__":
    helper = C2TNetHelper()

    img = cv2.imread("test1.png")
    out = helper.infer_single(img)
    print(out)
