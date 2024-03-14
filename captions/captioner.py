from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

model_path = 'Your_download_path'
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

templates = [
    "Wearing [clothing description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Sporting [hair description], the [person/woman/man] is dressed in [clothing description] and is carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "Carrying [belongings description], the [person/woman/man] is dressed in [clothing description] and [footwear description].",
    "In [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "Carrying [belongings description], the [person/woman/man] is wearing [clothing description] and [footwear description].",
    "In [clothing description] and [accessory description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is dressed in [clothing description] and [accessory description].",
    "Sporting [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description] and [accessory description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "In [clothing description] and [accessory description], the [person/woman/man] also has [hair description].",
    "In [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "With [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Wearing [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] sports [hair description] and is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Wearing [footwear description], [accessory description], [clothing description], the [person/woman/man] is also carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Dressed in [footwear description], [accessory description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description].",
    "In [accessory description], [footwear description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "Sporting [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "The [person/woman/man] is seen in [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] can be spotted wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is dressed in [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is attired in [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description].",
    "Dressed in [accessory description], [clothing description], [footwear description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description]."
]  
att = ['clothing','shoes','hairstyle','gender','belongings']

text = f'Write a description about the overall appearance of the person in the image, including the attributions: {att[0]}, {att[1]}, {att[2]}, {att[3]} and {att[4]}. If any attribute is not visible, you can ignore. Do not imagine any contents that are not in the image.'
text = f'Generate a description about the overall appearance of the person, including the {att[0]}, {att[1]}, {att[2]}, {att[3]} and {att[4]}, in a style similar to the template:"{temp}". If some requirements in the template are not visible, you can ignore. Do not imagine any contents that are not in the image.'
query = tokenizer.from_list_format([
                                    {'image': './figures/framework.png'}, # Either a local path or an url
                                    {'text': text},]
                                    )
caption, history = model.chat(tokenizer, query=query, history=None)
print(query)