# -*- coding: utf-8 -*-
# @Time : 2023/6/15 下午5:31
# @Author : zhenglinghan
# @Email : zhenglinghan.zlh@antgroup.com
# @File : run.py.py
# @Project : LLM

# Early load config to replace attn if needed
from arg_parser import get_config

ft_config = get_config()
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from utils.tricks.peft_tuners_lora_monkey_patch import replace_peft_model_with_gptq_lora_model

replace_peft_model_with_gptq_lora_model()

if ft_config.flash_attention:
    from utils.tricks.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()
elif ft_config.xformers:
    from utils.tricks.llama_attn_hijack_xformers import hijack_llama_attention

    hijack_llama_attention()

import autograd_4bit

if ft_config.backend.lower() == 'triton':
    autograd_4bit.switch_backend_to('triton')
else:
    autograd_4bit.switch_backend_to('cuda')

import peft
import peft.tuners.lora

import wandb
import torch
import transformers
from autograd_4bit import load_llama_model_4bit_low_ram, load_llama_model_4bit_low_ram_and_offload
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel, set_peft_model_state_dict

# ! Config
import train_data

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

if ft_config.mbatch_size > ft_config.batch_size:
    raise Exception('batch_size need to be larger than mbatch_size.')

# Load Basic Model
max_memory_gpu = ft_config.max_memory
max_memory_cpu = ft_config.max_memory_cpu

# example : max_memory = {0: '18Gib', 1: '18Gib', 'cpu': '48Gib'}
n_gpus = torch.cuda.device_count()
print(f"n_gpus: {n_gpus}")
max_memory_ = f'{max_memory_gpu}Gib'
max_memory = {i: max_memory_ for i in range(n_gpus)}
max_memory['cpu'] = f'{max_memory_cpu}Gib'

model, tokenizer = load_llama_model_4bit_low_ram_and_offload(ft_config.llama_q4_config_dir,
                                                             ft_config.llama_q4_model,
                                                             max_memory=max_memory,
                                                             # device_map=ft_config.device_map,
                                                             groupsize=ft_config.groupsize,
                                                             is_v1_model=ft_config.v1)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    device_map = ft_config.device_map
    if ft_config.ddp:
        device_map = {'': 0}
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {'': 0}
    print('Device map for lora:', device_map)
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map=device_map,
                                      torch_dtype=torch.float32, is_trainable=True)
    print(ft_config.lora_apply_dir, 'loaded')

    # Scales to half
    print('Fitting 4bit scales and zeros to half')
    for n, m in model.named_modules():
        if '4bit' in str(type(m)):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()

# Set tokenizer llama
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = 'left'

if not ft_config.skip:
    # Load Data
    data = None
    if ft_config.ds_type == "txt" and not ft_config.skip:
        #### LLaMa
        data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "alpaca" and not ft_config.skip:
        #### Stanford Alpaca-like Data
        data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "gpt4all" and not ft_config.skip:
        #### GPT4All Data
        data = train_data.TrainGPT4All(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "bluemoon" and not ft_config.skip:
        #### Blue Moon Data
        data = train_data.TrainBlueMoon(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    else:
        raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data(thd=ft_config.txt_row_thd, use_eos_token=ft_config.use_eos_token)
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing

        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

        # Count eval count for wandb
        if ft_config.val_set_size > 0:
            eval_count = 10
            eval_steps = max(
                ft_config.logging_steps,
                (len(data.train_data) + len(data.val_data)) // (eval_count * ft_config.mbatch_size)
            )
            print(f"Run eval every {eval_steps} steps")
        else:
            eval_steps = 0

        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=ft_config.mbatch_size,
            gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
            warmup_steps=ft_config.warmup_steps,
            optim="adamw_torch",  # paged_lion_8bit to get the lowest GPU memory occupy
            num_train_epochs=ft_config.epochs,
            learning_rate=ft_config.lr,
            fp16=True,
            logging_steps=ft_config.logging_steps,
            evaluation_strategy="steps" if eval_steps != 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if eval_steps != 0 else None,
            save_steps=ft_config.save_steps,
            output_dir=ft_config.lora_out_dir,
            save_total_limit=ft_config.save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ft_config.ddp else None,
            group_by_length=True,
            # deepspeed='ds.json' # deepspeed still buggy!
        )

        trainer = transformers.Trainer(
            model=model,
            train_dataset=data.train_data,
            eval_dataset=data.val_data,
            args=training_arguments,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        model.config.use_cache = False

        # Set Model dict
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

        # Set Verbose
        if ft_config.verbose:
            transformers.logging.set_verbosity_info()

        # Run Trainer
        with wandb.init(project="alpaca_lora_4bit") as run:
            if ft_config.resume_checkpoint:
                print('Resuming from {} ...'.format(ft_config.resume_checkpoint))
                state_dict_peft = torch.load(os.path.join(ft_config.resume_checkpoint, 'pytorch_model.bin'),
                                             map_location='cpu')
                set_peft_model_state_dict(model, state_dict_peft)
                trainer.train(ft_config.resume_checkpoint)
            else:
                trainer.train()

        # Restore old model state dict
        model.state_dict = old_state_dict

        print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')
