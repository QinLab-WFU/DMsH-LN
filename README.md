[DMsH-LN](https://www.sciencedirect.com/science/article/pii/S0925231224016011)

This paper is accepted for publication with Neucomputing.

## Training

### Processing dataset


### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128

### Citation
@ARTICLE{10530441,  
  author={Wu, Lei and Qin, Qibing and Hou, jinkui and Dai, jiangyan and Huang, Lei and Zhang, Wenfeng},  
  journal={Neurocomputing},  
  title={Deep multi-similarity hashing via label-guided network for cross-modal retrieval},  
  year={2024},  
  pages={5926-5939},  
  doi={https://doi.org/10.1016/j.neucom.2024.128830}}
