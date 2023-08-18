# Source Code for Eye-Tracking using Gaze-Data via [WebGazer.JS](https://github.com/brownhci/WebGazer) and [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)


### Installation

1. Install CUDA 11.8
2. (Windows) Make sure Visual Studio Build Tools (e.g., version 2022) are installed: [Link](https://code.visualstudio.com/docs/cpp/config-msvc)
3. We recommend using Anaconda with [Python 3.11.0](https://www.python.org/downloads/release/python-3110/) or higher
4. install `torch==2.0.1+cu118`

5. clone this repository: `git clone https://github.com/M-Colley/eye-tracking-pipeline.git`
6. run `pip install -r requirements.txt`


7. Follow installation guide of [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) without Docker (environment variables etc.)
8. We use SAM-HQ vit_h, download weights from [here](https://github.com/SysCV/sam-hq/issues/5) and put them into the root of our directory (`functions.py` looks for it there)
9. [Here](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py) you see how to switch the SAM model (should already be take care of in this work)


## Notes

- could be useful to use the Developer Command Prompt (unclear)
- Personalization: You will have to adapt your *custom prompt* for better results depending on your use case
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) states it needs supervision==0.6.0, but we upgraded to 0.13.0; this is also the case for the [Gradio App](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything)
- we also provide necessary functions to use 360 degree videos to work with yaw and pitch (`calculate_view(frame, yaw, pitch)`)
- Attention: the coding of the frames is highly important!
- the required quality of the detection can be altered by changing the values `box_threshold` and `text_threshold`. The higher the value, the fewer recognitions (true positives) but also less false positives you will find.
- Attention: `get_color_for_class` has to be adapted per use case