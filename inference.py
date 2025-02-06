import os
import sys
import shutil
import torch
from time import strftime

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


class SadTalker:
    def __init__(self, checkpoint_dir='./checkpoints', config_dir='src/config', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sadtalker_paths = init_path(checkpoint_dir, config_dir)

        # Load models
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

    def infer(self, source_image, driven_audio, output_dir='./results'):
        save_dir = os.path.join(output_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        # Step 1: Preprocess Image
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
        source_image, save_dir, source_image_flag=True, pic_size=256
        )
        if first_coeff_path is None:
            raise ValueError("Failed to extract 3DMM coefficients.")

        # Step 2: Audio-to-Coefficients
        batch = get_data(first_coeff_path, driven_audio, self.device, ref_eyeblink_coeff_path=None, still=False)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style=0)

        # Step 3: Animation
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, driven_audio, 
            batch_size=1, input_yaw_list=None, input_pitch_list=None, input_roll_list=None
        )
        
        result = self.animate_from_coeff.generate(data, save_dir, source_image, crop_info)
        output_video = save_dir + '.mp4'
        shutil.move(result, output_video)

        return output_video