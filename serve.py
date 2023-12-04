import gradio as gr
import re
import numpy as np
import torch
load = True
if load:
    from instructany2pix import InstructAny2PixPipeline
    pipe = InstructAny2PixPipeline()
    pipe.pipe.scheduler = pipe.pipe_inversion.scheduler
else:
    def pipe(*args,**kwargs):
        return None,None,None
VALID_MARKS = ['[image1]','[image2]','[image3]','[audio1]','[audio2]','[audio3]']
def run(image_input1,image_input2,image_input3,
        audio_input1,audio_input2,audio_input3,instruction,
        alpha,h0,h1,h2,norm,refinement,num_steps=50,seed=0,mode='default'):
    num_steps = int(num_steps)
    print(image_input1,image_input2,image_input3,audio_input1,audio_input2,audio_input3,
          instruction)
    matches = re.compile('\[[^\]]+\]').findall(instruction)
    mm_data_c = [image_input1,image_input2,image_input3,
        audio_input1,audio_input2,audio_input3]
    print(matches)
    if len(np.unique(matches)) != len(matches):
        return 'Error:Duplicate inputs',None
    mm_data = []
    for z in matches:
        if z  not in VALID_MARKS:
            return f'Error:Invalid inputs {z}',None
        idx = VALID_MARKS.index(z)
        if mm_data_c[idx] is None:
            return f'Error: Inputs {z} is referenced but not provided',None
        dtype = 'image' if 'image' in z else 'audio'
        payload = {"type": dtype, "fname": mm_data_c[idx], }
        mm_data.append(payload)
        instruction = instruction.replace(z,'<video>')
    
    print(mm_data,instruction)
    seed = int(seed)
    torch.manual_seed(seed)
    res0,res,output_caption = pipe(instruction,mm_data,
                                   alpha = alpha,h=[h0,h1,h2],
                                   norm=norm,refinement=refinement,num_inference_steps=num_steps,diffusion_mode=mode)
    return output_caption,res

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
examples = [
        [
            "assets/demo/an antique shop.jpg",None,None,
            "assets/demo/clock ticking.wav",None,None,
            'add [audio1] to [image1]',
            1.0,0.4,0.6,0.4,
            20,0.3,25,0
        ],
]

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("InstructAny2Any")  
        gr.Markdown("Input Image")  
        with gr.Row():
            image_input1 = gr.Image(type='filepath')
            image_input2 = gr.Image(type='filepath')
            image_input3 = gr.Image(type='filepath')
        gr.Markdown("Input Audio")  
        with gr.Row():
            audio_input1 = gr.Audio(type='filepath')
            audio_input2 = gr.Audio(type='filepath')
            audio_input3 = gr.Audio(type='filepath')
        gr.Markdown("Instruction")  
        with gr.Row():
            instruction = gr.Textbox()
        with gr.Row():
            alpha = gr.Slider(minimum=0.0,maximum=1.0,value=1.0,step=0.05,label='alpha')
            refinement = gr.Slider(minimum=0.0,maximum=1.0,value=0.3,step=0.1,label='refinement')
            seed =  gr.Slider(minimum=0.0,maximum=4096.0,value=0,step=1.0,label='seed')
        with gr.Row():
            norm = gr.Slider(minimum=0.0,maximum=20.0,value=20.0,step=1.0,label='norm')
            num_steps = gr.Slider(minimum=10.0,maximum=50.0,value=25.0,step=1.0,label='steps')
            h0 = gr.Slider(minimum=0.0,maximum=3.0,value=0.4,step=0.05,label='h0')
            h1 = gr.Slider(minimum=0.0,maximum=3.0,value=0.6,step=0.05,label='h1')
            h2 = gr.Slider(minimum=0.0,maximum=3.0,value=0.4,step=0.05,label='h2')
        with gr.Row():
            mode = gr.Dropdown(
                choices=['ipa','ipa_lcm','default']
            )
        with gr.Row():
            launch_button = gr.Button("Run")
        with gr.Row():
            out_text = gr.Text(interactive=False)
        with gr.Row():
            out_img = gr.Image(interactive=False)
        with gr.Row():
            gr.Examples(
            examples=examples,
            inputs=[
            image_input1,image_input2,image_input3,
            audio_input1,audio_input2,audio_input3,instruction,
             alpha,h0,h1,h2,norm,refinement,num_steps,seed
            ],
            # outputs=result,
            # fn=process_example,
            # cache_examples=CACHE_EXAMPLES,
            )
        launch_button.click(run,inputs=[
            image_input1,image_input2,image_input3,
            audio_input1,audio_input2,audio_input3,instruction,
             alpha,h0,h1,h2,norm,refinement,num_steps,seed,mode
        ],outputs=[out_text,out_img],api_name='run')
    demo.queue(max_size=20).launch(show_api=False,share=True,server_name="0.0.0.0")