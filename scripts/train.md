# Finetuning on Custom Data

You should prepare the following

### Preparing Edit Instructions

You should prepare a `instruction.json` which contains of list of image editing instructions. An example from MM-Inst datsset can be seen below

```
[
 {
  "task": "any2any",
  "base": "[a large cypress tree stands in the middle of a swamp]",
  "result": "an image of a large cypress tree stands in the middle of a swamp with an alligator in the foreground",
  "conversations": [
   {
    "from": "human",
    "value": "add [an image of a alligator] to [a large cypress tree stands in the middle of a swamp]"
   },
   {
    "from": "gpt",
    "value": "Absolutely, its,<base>[a large cypress tree stands in the middle of a swamp]<im_gen>[an image of a large cypress tree stands in the middle of a swamp with an alligator in the foreground]"
   }
  ]
]
```

### Preparing Media

`base` field should contatin the caption of the input image, `result` field should contatin the caption of the output image. Instruction should be formulated as conversation rounds in `conversations`, where multi-modal inputs are demarked by their caption using `[]`

Next, you need to prepare a collate all the media. Suppose you have file structure 

```
<root of your media files>
- data
---- 1.jpg # a large cypress tree stands in the middle of a swamp
---- 2.jpg # an image of a alligator
---- 3.jpg # an image of a large cypress ... with an alligator in the foreground
---- 4.mp3 # some audio used in other tasks
```


Next, you need to run `scripts/data_preparation.py` and encode the features of these media. You should obtain

```
<root of your media files>
- data
---- 1.jpg
---- 2.jpg 
---- 3.jpg
---- 4.mp3 
---- 1.jpg.npz
---- 2.jpg.npz
---- 3.jpg.npz
---- 4.mp3.npz
```

Now you should create a `media.json`, which contains a dictionary of caption-file mappings. The key of this dictionary should be the string used in the instrucion, and the value is a dictionary of the form ` {"fpath": "relative/path/to/npz", "key": "clip"}`

In the above example, you would have something like

```
{
    "a large cypress tree stands in the middle of a swamp": {
        "fpath":"data/1.jpg.npz",
        "key": "clip"
    },
    ...
}
```

Finally, you should replace the path arguments in `scripts/train.sh`  as follows 

```
DATA=<path to instruction data>
PRETRAINED_CKPT=<path to pretrained ckpt>
OUTPUT_DIR=<your out put dir>
IMAGE_ROOT=<root folder of media file>
MEDIA_MAP=<path to your media file>
```

After everything is ready, launch `bash scripts/train.sh` to launch training. 


# Finetuning on MM-Inst

Unfortunatetely, part of the dataset used in our work comes from WebVid (https://github.com/m-bain/webvid), which is no longer available as Shutterstock.com has send a cease and desist request to the WebVid authors. Hence, we also cannot redistribute that part of the dataset. We also have similar issues with LAION, Audioset. Hence, we are unable to share all raw media. 


Our data release at Huggingface [Link](https://huggingface.co/datasets/jacklishufan/MM-Inst-Train/tree/main) conatins the following elements. To better understand the use of each file, we strong suggest you read the previous section "Finetuning on Custom Data" first:

```
out_any_v5_music_plus.json # image editing instruction 
media_map.json # media map from caption to media, but some path is missing 
coco.tar media from COCO dataset
mTAT.tar media from MagnaTagATune dataset https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
instp2pdata.tar media from instructpix2pix dataset
```

The missing files are from LAION, AudioSet and WebVid dataset.

For AudioSet, it is possible to download the source video from youtube. For example, if you see keys like `[--4iCrbXlas]-[30-40].mp3.npz`in `media_map.json`, you can recover missing media by extract the audio of Youtube Video with ID `--4iCrbXlas` between 30-40 seconds. We used [youtube-dl](https://github.com/ytdl-org/youtube-dl) to download the data. Unfortunately, it appears that this tool has been blocked by Youtube and is no longer working.


