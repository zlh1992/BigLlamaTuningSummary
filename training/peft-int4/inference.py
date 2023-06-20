# -*- coding: utf-8 -*-
# @Time : 2023/6/20 上午11:03
# @Author : zhenglinghan
# @Email : zhenglinghan.zlh@antgroup.com
# @File : inference.py
# @Project : BigLlamaTuningSummary

import os
import sys
import time
import torch

# import matmul_utils_4bit
# matmul_utils_4bit.faster = False

from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear, \
    load_llama_model_4bit_low_ram_and_offload

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from utils.tricks.peft_tuners_lora_monkey_patch import replace_peft_model_with_gptq_lora_model

replace_peft_model_with_gptq_lora_model()
from utils.tricks.llama_attn_hijack_xformers import hijack_llama_attention

hijack_llama_attention()
# from utils.tricks.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()

config_path = '/data/Workspace/TEMP/DATA/ckpt/Neko-Institute-of-Science-LLaMA-65B-4bit-128g/'
model_path = '/data/Workspace/TEMP/DATA/ckpt/Neko-Institute-of-Science-LLaMA-65B-4bit-128g/llama-65b-4bit-128g.safetensors'

n_gpus = torch.cuda.device_count()
print(f"n_gpus: {n_gpus}")
max_memory_gpu = 23
max_memory_cpu = 48
max_memory_ = f'{max_memory_gpu}Gib'
max_memory = {i: max_memory_ for i in range(n_gpus)}
max_memory['cpu'] = f'{max_memory_cpu}Gib'

lora_path = "../sft65b/checkpoint-500/"

model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, max_memory=max_memory, groupsize=128,
                                                 is_v1_model=False)

from peft import PeftModel

model = PeftModel.from_pretrained(model, lora_path)

print('Fitting 4bit scales and zeros to half')
model.half().eval()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.is_v1_model:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

print('Apply AMP Wrapper ...')
from amp_wrapper import AMPWrapper

wrapper = AMPWrapper(model)
wrapper.apply_generate()


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {data_point['input']}
    ### Response:"""


while 1:
    inputs = input("User:")
    prompt = {'input': inputs}
    prompt = generate_prompt(prompt)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.cuda() for k, v in batch.items()}
    start = time.time()
    with torch.no_grad():
        generated = model.generate(inputs=batch["input_ids"],
                                   do_sample=True, use_cache=True,
                                   repetition_penalty=1.3,
                                   max_new_tokens=256,
                                   temperature=0.3,
                                   top_p=0.8,
                                   top_k=40,
                                   return_dict_in_generate=True,
                                   output_attentions=False,
                                   output_hidden_states=False,
                                   output_scores=False)
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    end = time.time()
    print("Assistant:", result_text.split("### Response:")[-1].split("###")[0])
    print(end - start)
