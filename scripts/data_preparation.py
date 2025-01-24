# This file contains example script to pre-process image and audio data
from imagebind import imagebind_model
from imagebind import data as image_bind_data
from imagebind.models.imagebind_model import ModalityType
import numpy as np
def process_one_audio(model_imb,fpath):
    audio_paths = [fpath,]
    save_path = fpath + '.npz'
    inputs = dict(audio=image_bind_data.load_and_transform_audio_data(audio_paths,'cuda'))
    out_model_imb = model_imb(inputs)['audio']
    np.savez_compressed(
            save_path,
            #state=y[0].cpu().numpy(),
            clip=out_model_imb.cpu().numpy(),
        )
    
def process_one_image(model_imb,fpath):
    image_paths = [fpath,]
    save_path = fpath + '.npz'
    inputs = dict(vision=image_bind_data.load_and_transform_vision_data(image_paths,'cuda'))
    out_model_imb = model_imb(inputs)['vision']
    np.savez_compressed(
            save_path,
            clip=out_model_imb.cpu().numpy(),
        )
    
if __name__ == '__main__':
    import argparse,glob,os
    model_imb = imagebind_model.imagebind_huge(pretrained=True)
    model_imb.eval()
    model_imb.to('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',type=str)
    args = parser.parse_args()
    audio_files = glob.glob(os.path.join(args.path,'*.mp3'))
    image_files = glob.glob(os.path.join(args.path,'*.jpg'))
    for fpath in image_files:
        process_one_image(fpath)
    for fpath in audio_files:
        process_one_audio(fpath)
    