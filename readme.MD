# InstructAny2Pix: Flexible Visual Editing via Multimodal Instruction Following


    [DEMO](http://170.106.137.89:8888)
    [Project Page](http://homepage.jackli.org/projects/instructany2pix.html)
    [Paper](http://homepage.jackli.org/projects/Instructany2Pix.pdf)
    [HuggingFace](https://huggingface.co/jacklishufan/instructany2pix/tree/main)

PyTorch implementation of InstructAny2Pix: Flexible Visual Editing via Multimodal Instruction Following

Shufan Li, Harkanwar Singh, Aditya Grover 

University of California, Los Angeles


![Img](assets/appendix1-3.png)
## Installation

```
conda create --name instructany2pix python=3.10
conda activate instructany2pix
conda install ffmpeg
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
pip install git+https://github.com/facebookresearch/ImageBind.git --no-deps
```

## TODO
- [] Release more checkpoints
- [] Release Training Code, Data.

## Running

To serve gradio app, run
```
python serve.py
```

Alternatively, one can check `demo.ipynb` for a notebook demo

## Q&A

### Where can I find checkpoints
See Huggingface Link at the top

### The edit is not what I want

We are working on improving the roboustness of the method. At the moment, there are ways you can contorl the output.

#### It changes little
Try to Increase `alpha`, `norm`, decrease `h_0`

#### It changes too much
Try to decrese `alpha`, `norm`, decrese `h_0`

#### Low quality outputs
Try to Increase `refinement`, `h_2`, `steps`

#### Content of image is not intended
Try to change `seed` and increase `h_1`

