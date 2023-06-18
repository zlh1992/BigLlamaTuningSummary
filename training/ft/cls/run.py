# -*- coding: utf-8 -*-
# @Time : 2023/6/18 下午9:20
# @Author : zhenglinghan
# @Email : zhenglinghan.zlh@antgroup.com
# @File : run.py
# @Project : LLM


# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import time

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from utils.tricks.llama_attn_hijack_xformers import hijack_llama_attention

hijack_llama_attention()
major, minor = torch.cuda.get_device_capability()
if major >= 8:
    print('Your GPU supports bfloat16 and flash-attention, you can accelerate training with the argument --bf16')
    from utils.tricks.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()

from transformers import (
    SchedulerType,
    get_scheduler,
    AutoTokenizer,
    LlamaTokenizer,
    default_data_collator
)

from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `2,4,4`'
                             'will use 60% of data for phase 1, 20% for phase 2'
                             'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_tdsrain_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    parser.add_argument('--log_step',
                        type=int,
                        default=10,
                        help="local_rank for distributed training on gpus")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="query",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.bos_token = "<s>"
    tokenizer.eor_token = "</s>"
    tokenizer.pad_token = '<unk>'

    cls_model = create_critic_model(args.model_name_or_path,
                                    tokenizer,
                                    ds_config,
                                    args.num_padding_at_beginning,
                                    disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        print("lora using!")
        cls_model = convert_linear_layer_to_lora(cls_model,
                                                 args.lora_module_name,
                                                 args.lora_dim)
        if args.only_optimize_lora:
            cls_model = only_optimize_lora_parameters(cls_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def distributed_concat(tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        losses = 0
        labels = []
        preds_acc = []
        step = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = to_device(batch, device)
                outputs = model(**batch)
                pred_acc = F.softmax(outputs['logits'].float(), dim=1)[:, 1]  # softmax for classes
                preds_acc.append(pred_acc.reshape(-1, ))  # tensor
                labels.append(batch['labels'].reshape(-1, ))  # tensor

                loss = outputs['loss']
                losses += loss.float()
        losses = losses / (step + 1)
        predictions = distributed_concat(torch.concat(preds_acc, dim=0),
                                         len(eval_dataloader)).cpu().numpy().tolist()
        labels = distributed_concat(torch.concat(labels, dim=0),
                                    len(eval_dataloader)).cpu().numpy().tolist()
        auc_model = roc_auc_score(labels, predictions)
        model.train()
        return losses, auc_model

    # Split weights in two groups, one with weight decay and the other not.
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    #     cls_model, args.weight_decay)

    # Split weights in two groups, one with weight decay and the other not.
    def create_optimizer_grouped_parameters(model, model_lr, weight_decay, adam_epsilon=1e-8, use_bertadam=False):
        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = []
        other_param_optimizer = []

        names_other = []
        for name, para in model.named_parameters():
            if 'classifier' not in name and 'NextVLAD' not in name and 'score' not in name and 'v_head' not in name:  # 需要注意修改 过滤出bert的参数
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))
                names_other.append(name)
        print("names_other:", names_other)
        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': model_lr['bert']},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': model_lr['bert']},
            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': model_lr['classifier']},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': model_lr['classifier']},
        ]
        return optimizer_grouped_parameters

    model_lr = {'bert': args.learning_rate, 'classifier': 1e-3}
    optimizer_grouped_parameters = create_optimizer_grouped_parameters(cls_model, model_lr, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    cls_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=cls_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        cls_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    eval_loss, eval_auc = evaluation_reward(cls_model, eval_dataloader)
    print_rank_0(
        f"eval_loss (lower is better) : {eval_loss}, eval_auc (higher is better) : {eval_auc}",
        args.global_rank)

    start_log_time = time.time()
    patience = 2
    now_patience = 0
    max_eval_auc = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        cls_model.train()
        mean_loss = 0
        step = 0
        training_step_losses = []
        self_training_step_losses = []
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = cls_model(**batch, use_cache=False)
            loss = outputs["loss"]
            cls_model.backward(loss)
            cls_model.step()
            mean_loss += loss.item()
            training_step_losses.append(loss)
            if step % args.log_step == 0:
                end_log_time = time.time()
                log_time = end_log_time - start_log_time
                _loss = sum(training_step_losses) / len(training_step_losses)
                _self_loss = sum(self_training_step_losses) / len(training_step_losses)
                _log_step = (epoch * len(train_dataloader)) + step + 1
                _speed = (log_time) / ((epoch * len(train_dataloader)) + step + 1)
                _train_schedule = ((epoch * len(train_dataloader)) + step + 1) / (
                        args.num_train_epochs * len(train_dataloader))
                _all_to_consume = (log_time) / (((epoch * len(train_dataloader)) + step + 1) / (
                        args.num_train_epochs * len(train_dataloader)))
                _estimated_to_consume = ((log_time) / (((epoch * len(train_dataloader)) + step + 1) / (
                        args.num_train_epochs * len(train_dataloader)))) * (1 - (
                        ((epoch * len(train_dataloader)) + step + 1) / (
                        args.num_train_epochs * len(train_dataloader))))
                s_print = f"epoch {epoch} step {step} train loss {_loss}, log_step {_log_step}, speed {_speed}, train schedule {_train_schedule}, all to consume {_all_to_consume}, estimated to consume {_estimated_to_consume}, rank {args.global_rank}"
                print_rank_0(s_print, args.global_rank)
        print_rank_0(
            f"Epoch {epoch + 1}/{args.num_train_epochs} with loss {mean_loss / (step + 1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} *****",
            args.global_rank)
        eval_loss, eval_auc = evaluation_reward(cls_model, eval_dataloader)
        print_rank_0(
            f"eval_loss (lower is better) : {eval_loss}, eval_auc (higher is better) : {eval_auc}",
            args.global_rank)
        cls_model.tput_timer.update_epoch_count()
        if eval_auc > max_eval_auc:
            # save sota
            print_rank_0("save sota!", args.global_rank)
            if args.output_dir is not None:
                # args.output_dir = "./output"
                print_rank_0(f'saving model ...{args.output_dir}', args.global_rank)
                if args.lora_dim > 0:
                    cls_model = convert_lora_to_linear_layer(cls_model)
                if args.global_rank == 0:
                    save_hf_format(cls_model, tokenizer, args)
                if args.zero_stage == 3:
                    # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                    save_zero_three_model(cls_model,
                                          args.global_rank,
                                          args.output_dir,
                                          zero_stage=args.zero_stage)
            max_eval_auc = eval_auc
            now_patience = 0
        else:
            now_patience += 1
            if now_patience == patience:
                break
    end_log_time = time.time()
    print_rank_0(f'train end time use: {end_log_time - start_log_time}', args.global_rank)


if __name__ == "__main__":
    main()
