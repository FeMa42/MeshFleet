# Loading and Processing of Objaverse XL Dataset

## Installation

You need to download an install Blender:

```bash
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && \
  tar -xf blender-3.2.2-linux-x64.tar.xz && \
  rm blender-3.2.2-linux-x64.tar.xz
```

See [Objaverse-XL Rendering Script](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering) for more details.

## Usage

You have to set you blender path as well as the location of where you want to save the rendered images in 'setting.py'.

After that you can download and render multiple views of 3D objects from the Objaverse XL dataset using Blender with 'objaverse_xl_batched_renderer.py'. Example call: 

```bash
python objaverse_xl_batched_renderer.py --sample_size 10 --n_processes 10 --batch_size 10 --gpu_batch_size 10
```
