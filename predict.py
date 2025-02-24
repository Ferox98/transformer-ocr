import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate, CustomDataset
from datasets import load_metric
import pandas as pd

def compute_metrics(df):
    
    print(df.head(5))
    target = df[0].to_list()
    labels = df[1].to_list()
    pred=[]
    print(f'some image dirs: {target[:10]}')
    print('entring loop ...')
    print(len(target))
    for i,img in enumerate(target):
        print(i)
        x = collate.ready_image(Path(img))
        pred.append(model.predict(x))
    print('loading metrics')   
    cer_metric = load_metric("cer")
    # bleu_metric = load_metric('bleu')
    print('computing cer')
    cer = cer_metric.compute(predictions=pred, references=labels)
#     bleu = bleu_metric.compute(predictions=list(pred_str), references=list(list(label_str)))

    return {"cer":cer}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--target", "-t", type=str, required=True,
                        help="OCR target (image or directory)")
    parser.add_argument("--load_tokenizer", "-tk", type=str, required=True,
                        help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Load model weight in checkpoint")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load
    tokenizer = load_tokenizer(cfg.load_tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer)
    saved = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = CustomCollate(cfg, tokenizer=tokenizer,is_train=False)

    df = pd.read_csv(cfg.val_data, sep="\t", header=None)

    target = Path(cfg.target)
    if target.is_dir():
        target = list(target.glob("*.jpeg")) + list(target.glob("*.png"))
    else:
        target = [target]

    for image_fn in target:
        print(image_fn)
        start = time.time()
        x = collate.ready_image(image_fn)
        print("[{}]sec | {} : {}".format(time.time()-start, image_fn, model.predict(x)))

    # print(compute_metrics(df))


