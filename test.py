import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate, CustomDataset
from datasets import load_metric
import pandas as pd
from torch.utils.data import DataLoader

def compute_metrics(model, test):

    cer_metric = load_metric("cer")
    cer=0
    print('entring loop ...')
    for data in test:
            features, labels = data
            features, labels = features, tokenizer.decode(labels[0])
            print(model(features))
            # print('computing cer')
            # cer += cer_metric.compute(predictions=model.predict(features), references=labels)

    
    # bleu_metric = load_metric('bleu')
   
#     bleu = bleu_metric.compute(predictions=list(pred_str), references=list(list(label_str)))

    return True

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

    tokenizer = load_tokenizer(cfg.load_tokenizer)
    val_set = CustomDataset(cfg, cfg.val_data)
    val_collate = CustomCollate(cfg, tokenizer, is_train=False)
    val_dataloader = DataLoader(val_set, batch_size=128,
                                  num_workers=16, collate_fn=val_collate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load
    tokenizer = load_tokenizer(cfg.load_tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer)
    saved = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = CustomCollate(cfg, tokenizer=tokenizer,is_train=False)

    df = pd.read_csv(cfg.val_data, sep="\t", header=None)

    target = Path(cfg.target)
    # if target.is_dir():
    #     target = list(target.glob("*.jpg")) + list(target.glob("*.png"))
    # else:
    #     target = [target]

    # for image_fn in target:
    #     start = time.time()
    #     x = collate.ready_image(image_fn)
    #     print("[{}]sec | {} : {}".format(time.time()-start, image_fn, model.predict(x)))

    print(compute_metrics(model, val_dataloader))


