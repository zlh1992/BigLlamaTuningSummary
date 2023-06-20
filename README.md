# Llama模型训练框架汇总项目
针对Llama模型包含了一些常见训练技巧，分为sft, cls两种任务。项目的宗旨就是 大家不要再玩7b的Llama了！通过这个项目可以比较顺畅的使用标准数据集训练大模型与一些分类，回归的NLP任务。follow了最新的训练框架，提供最为low cost的训练脚本,实现消费级显卡训练7b+的模型。
sft对应dschat的step1 cls任务作为大模型做回归，分类任务，可以扩展到dschat的step2 单机多卡。多重框架的数据流做了整合。Llama统一按照左padding。

每种任务的训练提供qlora、全量、peft-int4 三种模式：

qlora模式最为low cost 采用accelerate+int4+qlora训练只支持单机单卡单机多卡，可用于快速训练小模型验证数据（in4精度下速度>float16>in8），也可分拆模型到多卡训练大模型（多卡总计利用率=1卡，Memory Bubbles比较大）。

全量模式采用float16的deepspeed也可以使用float16的lora，也可以使用offload+deepspeed 实现low cost的全量训练。

还提供了一种accelerate+peft in4的sft训练框架，可以直接对qlora量化后的模型进行int4训练，该框架代码十分值得学习可以扩展到int2训练进一步实现low cost的目标。


# 安装
pip install -r requirements.txt
pip install --force git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit
pip install git+https://github.com/johnsmith0031/alpaca_lora_4bit@winglian-setup_pip

# 运行
各个文件下bash run.sh 默认环境为4卡v100 32GB

peft-int4:
seqlen=2048 bs=1 自有数据集

{'loss': 5.3615, 'learning_rate': 9.6e-05, 'epoch': 0.02}                                                                                                                                       
{'loss': 1.035, 'learning_rate': 9.804081632653061e-05, 'epoch': 0.04}                                                                                                                          
{'loss': 1.0066, 'learning_rate': 9.6e-05, 'epoch': 0.06}                                                                                                                                       
{'loss': 0.9869, 'learning_rate': 9.395918367346939e-05, 'epoch': 0.08}
 10%|██████████████▎    | 495/5000 [21:21:16<194:41:15, 155.58s/it]
 10w数据可以看到是很慢的

# 显存占用
ft 7b模型24gb显存

peft-int4 支持单机多卡24gb*2即可训练65b llama

qlora 支持单机多卡24gb*2即可训练65b llama， 7b模型最小8g显存 


4卡v100 peft-int4 llama65b:
Total 7860.77 Gib VRAM used.
Loaded the model in 33.07 seconds.
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.116.04   Driver Version: 525.116.04   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:3D:00.0 Off |                    0 |
| N/A   44C    P0    41W / 250W |  23068MiB / 32768MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  On   | 00000000:41:00.0 Off |                    0 |
| N/A   50C    P0   260W / 250W |  27964MiB / 32768MiB |     97%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  On   | 00000000:B1:00.0 Off |                    0 |
| N/A   41C    P0    41W / 250W |  27966MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  On   | 00000000:B5:00.0 Off |                    0 |
| N/A   41C    P0    45W / 250W |  26696MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

# 内存占用
目前65b模型训练需要64GB+的内存 服务器128GB会比较稳

# Tips
amper架构的显卡才能使用flash attention 以及bf16计算(A100)

不建议int4训练7b模型的sft任务 较大的模型抗精度降低的效力才会更好

从训练速度看，int4 lora>=float16(全量/lora)>>int8 lora（huggingface对int8的支持不太好）

同样是int4 gptq精度>bitsandbytes


# TODO
数据集收集:
补充质量较高的开源数据集与清洗脚本

前处理:
code数据清洗

训练:
完善qlora+deepspeed
完善peft-int4+deepspeed
follow peft+int2训练
补充reward标准数据集的训练case
补充多种ppo框架

推理:
一些serving example
codeT(https://github.com/microsoft/CodeT)
NBCE(https://github.com/bojone/NBCE)
fast transformer

部署:
webgui(https://github.com/lm-sys/FastChat)

# 参考项目
https://github.com/lm-sys/FastChat
https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
https://github.com/johnsmith0031/alpaca_lora_4bit
https://github.com/artidoro/qlora

# 合作伙伴
https://github.com/yrqUni
