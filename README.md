# llama2-7b-miniguanaco-dpo
**Two-stage fine-tuning model (SFT+DPO) based on llama2-7b-chat**
- sft_model:[ccccai/llama-2-7b-guanaco-3.5k](https://huggingface.co/ccccai/llama-2-7b-guanaco-3.5k)
- sft+dpo_model:[ccccai/llama-2-7b-guanaco-3.5k-dpo](https://huggingface.co/ccccai/llama-2-7b-guanaco-3.5k-dpo)
<br>To develop a chatbot with a dedicated task, this project uses llama2-7b-chat as the basic model, and performs two-stage fine-tuning of SFT and DPO on it to enhance its overall performance and make its output as consistent as possible with humans.
##### 技术路线
![技术路线.jpeg](https://github.com/ccccai239/llama2-7b-miniguanaco-dpo/blob/main/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF.jpeg)
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
* 运行sft.py进行SFT阶段，可以修改model_name,dataset_name实现不同LLM的sft微调
* 由于内存有限，可采用LoRA或QLoRA技术进行高效参数微调，实验就运用了QLoRA参数微调技术，将lora注意力维度lora_rank设置为64，缩放参数为16，并使用 NF4 类型直接以 4 位精度加载 Llama 2 模型（Llama-2-7b-chat-hf）以及分词器
* QLoRA参数可以调整
* 本实验在miniguanaco数据集（仅取3500个样本）进行sft，微调得到的模型已上传至huggingface&emsp;[ccccai/llama-2-7b-guanaco-3.5k](https://huggingface.co/ccccai/llama-2-7b-guanaco-3.5k)
```python
!python sft.py 
```
## RLHF-DPO
* RLHF指通过使用来自人类反馈的数据来训练模型，以改进其性能和输出质量
* DPO直接偏好优化，首次由斯坦福团队提出[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2305.18290.pdf)，它的创新点在于直接优化LM来对齐人类偏好，无需建模reward model和强化学习阶段
* 使用trl库中DPOTrainer组件实现，**trl库版本需>=0.7.0**
```python
!python dpo.py \
    --model_name_or_path="ccccai/llama-2-7b-guanaco-3.5k" #model_name可以替换
```
## 上传至huggingface
```python
!huggingface-cli login
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
```
<br>请注意，这里huggingface登陆的api token对应的应该是**WRITE**功能
## 调用模型-prompt训练专属chatbot
```python
from transformers import pipeline, set_seed

def generate_cot_response(prompt_template, input_data, model_name="ccccai/llama-2-7b-guanaco-3.5k-dpo"):
    # Combine the prompt template with the actual input data
    prompt = prompt_template.format(input_data)

    # Initialize the pipeline with the specified model
    generator = pipeline("text-generation", model=model_name)

    # Generate a response
    response = generator(prompt, max_length=150, num_return_sequences=1)

    return response[0]['generated_text']

# Example
prompt_template = "Let's think step by step to solve this: {}"
input_data = "If I have 3 apples and buy 5 more, how many do I have?"

response = generate_cot_response(prompt_template, input_data, model_name="ccccai/llama-2-7b-guanaco-3.5k-dpo") 
print(response)

```
