# -*- coding: utf-8 -*-
# @Time : 2023/6/18 下午9:28
# @Author : zhenglinghan
# @Email : zhenglinghan.zlh@antgroup.com
# @File : run_eval.py
# @Project : LLM


# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
print("start!")
import argparse
import os
import torch

import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='training_scripts/single_node/output_0605',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):
    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = "left"
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.pad_token_id = 0
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer


def prepare_singlesample(prompt,
                         tokenizer,
                         max_seq_len=512):
    chosen_token = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")  # 不填充
    # 默认left
    chosen_token['input_ids'] = chosen_token['input_ids'][:, -max_seq_len:]
    chosen_token['attention_mask'] = chosen_token['attention_mask'][:, -max_seq_len:]
    chosen_token.pop('token_type_ids')
    return chosen_token


def run_single_sample(prompts):
    args = parse_args()
    device = torch.device("cuda")
    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()
    # Run inference
    res = []
    labels = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            batch = prepare_singlesample(
                prompt['input'],
                tokenizer,
                max_seq_len=2048
            )
            batch = to_device(batch, device)
            outputs = rm_model(**batch)
            y_hat = F.softmax(outputs['logits'].float(), dim=1)[:, 1].cpu().numpy().tolist()
            res += y_hat
            if prompt['output'] != 'nan':
                labels.append(int(prompt['output']))
    if len(labels) > 0:
        print("check auc:", roc_auc_score(labels, res))
    return res


if __name__ == "__main__":
    import json

    with open("/ossfs/workspace/key_code_01_test_use_v0605.json", 'r') as f:
        prompts = json.load(f)
    res = run_single_sample(prompts)
    import joblib

    joblib.dump(res, "res.pkl")
