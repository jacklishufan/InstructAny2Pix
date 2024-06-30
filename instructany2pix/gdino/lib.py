import numpy as np
import cv2
import groundingdino.datasets.transforms as T
from PIL import Image, ImageFilter  
import os
from groundingdino.util.inference import load_model, predict, annotate
import torch
def load_image_and_transform(image_path):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = image_path.convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_mask(ph,boxes,phrases,predictor,i=0,d=40,e=10,b=0):
    base = np.zeros((1024,1024,3),dtype=np.uint8)
    #print(phrases,ph,np.array(phrases)==ph)
    zz = [(ph in x or x in ph) for x in phrases ]
    box = boxes[np.array(zz)][i]
    box = box * 1024
    box = box.int().numpy()
    pt1 = (box[0]-box[2]//2,box[1]-box[3]//2)
    pt2 = (box[0]+box[2]//2,box[1]+box[3]//2)
    box = np.array([*pt1,*pt2])
    base = cv2.rectangle(base, pt1,pt2, (255,255,0), thickness=-1)
    base = base[...,0]
    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box.reshape(1,4) ,
    multimask_output=False,
    )

    mask =  masks[0].astype(np.uint8)*255
    z=d
    kernel = np.ones((e, e), np.uint8)  # You can adjust the size of the kernel as needed
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((z, z), np.uint8)  # You can adjust the size of the kernel as needed
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = Image.fromarray(mask)
    if b > 0:
        #kernel_size = (b, b)
        mask = mask.filter(ImageFilter.GaussianBlur(radius = b)) 
        #mask = cv2.blur(mask, kernel_size)
    return mask


def build_segmentator(ckpt='./ckpts'):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_h"](os.path.join(ckpt,"gdino/sam_vit_h_4b8939.pth"))
    from segment_anything import SamPredictor
    sam.to(device='cuda')

    predictor = SamPredictor(sam)
    from groundingdino.util.inference import load_model, predict, annotate
    import cv2

    gdino = load_model(os.path.join(ckpt,"gdino/GroundingDINO_SwinT_OGC.py"), 
                       os.path.join(ckpt,"gdino/gdino.pth")
                       )
    return predictor,gdino

def subject_consistency(subjec_data,output_caption,img,sam,gdino,pipe_inpainting,subject_strength=0.7):
    TEXT_PROMPT = '. '.join(x[0] for x in subjec_data)
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    sam.set_image(np.array(img))
    image_source,transformed_image = load_image_and_transform(img)
    boxes, logits, phrases = predict(
        model=gdino,
        image=transformed_image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    subject = img
    # pipe_inpainting.set_ip_adapter_scale(0.5)
    for ph,emb in  subjec_data:
        ph = ph.replace('.','').replace("'s",'')
        msk = get_mask(ph,boxes,phrases,sam,i=0,d=40,b=20)
        print(f"Subject Embed: {emb.shape}")
        subject = pipe_inpainting.generate(
                        image=subject,
                        mask_image=torch.tensor(np.array(msk)),
                        pil_image=None,
                        strength=subject_strength,
                        #prompt='best quality, high quality',#+output_caption, 
                        #negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
                        clip_image_embeds_local=emb[None],
                        mode='local',
                        num_inference_steps=50,
                        # seed=0,
                        # guidance_scale=5,
                        scale=0.8,
                        )[0]
        
    return subject,annotated_frame