# llama2-7b-miniguanaco-dpo
**Two-stage fine-tuning model (SFT+DPO) based on llama2-7b-chat**

To develop a chatbot with a dedicated task, this project uses llama2-7b-chat as the basic model, and performs two-stage fine-tuning of SFT and DPO on it to enhance its overall performance and make its output as consistent as possible with humans.
#### 技术路线

## 安装环境
 ```python
!pip install -r requirements.txt
 ```
- python==3.10
- Pytorch==12.2
- transformers==4.31.0 
- trl==0.7.0
- GPU：A100，40G
## 数据集处理
#### [SFT Data](https://github.com/ccccai239/llama2-7b-miniguanaco-dpo/sft_data_pro.py)
<br>运行sft_data_pro.py文件格式化数据输入，将其统一成以下形式：
```
<s>[INST] <<SYS>>
System prompt
<</SYS>>

User prompt [/INST] Model answer </s>
```
<br>_因为本项目采用的是chat模型，若使用的是base模型，则不需要将input格式化为该模式_
#### [DPO_Data](https://huggingface.co/datasets/AlexHung29629/stack-exchange-paired-128K)
<br>_完整数据集为[lvwerra/stack-exchange-paired](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)_
<br>每个输入样本都由[prompt]+[chosen]+[rejected]三部分组成，每个Question分别对应两个问题，一个好答案（chosen）和一个坏答案（rejected），这个好坏的标注是人工处理，主要用于LLM的RLHF阶段，旨在与人类价值观对齐。
```python
{
            #"prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            #"chosen": samples["response_j"],
            #"rejected": samples["response_k"],
            "prompt":samples["prompt"],
            "chosen":samples["chosen"],
            "rejected":samples["rejected"]
        }
```
## 加载模型和数据集
<br>1. 加载基线模型[LlaMA2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```
<br>2. 加载数据集
```python
from datasets import load_dataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k")
```
## SFT

## RLHF-DPO
