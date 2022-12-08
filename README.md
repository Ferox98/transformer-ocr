# Transformers for ocr
[TrOCR](https://arxiv.org/abs/2109.10282) 
[swin-transformer](https://arxiv.org/abs/2103.14030) 

## Overview
In this project, we explored how Transformers can be applied to OCR. We experimented with a
pretrained image Transformer as encoder and a language modelling decoder. We also trained a
transformer with randomly initialized weights to compare the results and emphasize the advantage of
pretraining. Additionally, we saw the viability of using an efficient vision transformer architecture,
the Shifted Window Transformer (SWIN), as an encoder in an OCR transformer pipeline.

The model in this repository heavily relied on the swin transformer implmentation by [YongWookHa](https://github.com/YongWookHa/swin-transformer-ocr). 
we've added a preprocessing script for both IIIT-HWS and IAM datasets. We have also added a way to evaluate the model's performance based on the CER metric. 
We've also added a notebook to evaluate TrOCR baseline model on the IAM dataset. 
## Performance
We ran the experiments against the IAM dataset using pretrained model weights as well as randomly
initialized ones for both the baseline model and its SWIN variant.  We show
that the SWIN transformer with significantly fewer parameters can perform comparably with BEiT
transformer if pretrained sufficiently. We believe the results we obtained are promising enough to
warrant additional investigation into the use of efficient transformer models in OCR for low-resource
environments.

## Data
For pretraining
Download Ground truth files and IIIT-HWS image corpus from the iiit-dataset and extract in the dataset/iiit directory
For training
Download the data/lines.tgz and data/xml.tgz files from the IAM dataset page and extract to the dataset/IAM/lines and  dataset/IAM/labels directorys respectively
```bash
./dataset/
├─ iam/
│  ├─ labels
│  |	├─ a01-000u.xml
│  |	├─ ...
|  ├─ lines
|  |	├─ a01/
│  |	├─ ...
├─ iiit/
│  ├─ groundtruth
│  |	├─ IIIT-HWS-10K.mat
│  |	├─ IIIT-HWS-90K.mat
|  ├─ Images_90K_Normalized
|  |	├─ 1/
|  |    ├─ 2/
│  |	├─ ...
├─ train_iiit.txt
└─ test_iiit.txt
├─ train_iam_lines.txt
└─ test_iam_lines.txt


# in train.txt
path/to/image_1.jpg\tHello World.
path/to/image_2.jpg\tvision-transformer-ocr
...
```
## Prepare data 
```bash
python prepare_data.py -d path/to/data 
```


## Configuration
In `settings/` directory, you can find `default.yaml`. You can set almost every hyper-parameter in that file. Copy one and edit it as your experiment version. I recommend you to run with the default setting first, before you change it.

## Train-SWIN

```bash
python run.py --version 0 --setting settings/default.yaml --num_workers 16 --batch_size 128
```
you can also use the pretrained weights and the tokenier from [here](https://drive.google.com/drive/folders/11zfmHue5YtujEKNWRkpAZNYYRFfFiXMd?usp=sharing) to run inference (recomended since training takes around 3 hours.)
You can check your training log with tensorboard.  
```
tensorboard --log_dir tb_logs --bind_all
```
  

## Predict-SWIN
When the model finishes training, make predictions using by runing the predict script.

```bash  
python predict.py --setting <your_setting.yaml> --target <image_or_directory> --load_tokenizer <your_tokenizer_pkl> --checkpoint <saved_checkpoint>
```

## Train and inference-TrOCR

```bash
Open the trocr.ipynb notebook and run the cells
```

## Citations

```bibtex
@misc{liu-2021,
    title   = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
	author  = {Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
	year    = {2021},
    eprint  = {2103.14030},
	archivePrefix = {arXiv}
}

@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
