# InstructAny2Pix: Flexible Visual Editing via Multimodal Instruction Following

PyTorch implementation of InstructAny2Pix: Flexible Visual Editing via Multimodal Instruction Following

Shufan Li Harkanwar Singh   Aditya Grover 

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



