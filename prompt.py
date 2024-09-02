import os
import json

from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from huggingface_hub import InferenceClient

from preprocess import construct_prompt, construct_prompt_movie
from analyze import calculate_metrics
from helper import save_prompt_as_array, get_temperature


with open('API_KEYS.json', 'r') as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config['openai_key']
client = OpenAI()

os.environ["AZURE_INFERENCE_CREDENTIAL"] = config['azure_key']
client_azure = ChatCompletionsClient(
    endpoint='https://Phi-3-5-mini-instruct-irlfw.eastus.models.ai.azure.com',
    credential=AzureKeyCredential(config['azure_key'])
)

def prompt_chatgpt(model_name, conv_id, prompt_type = "context_only", dataset_name = "personachat", dataset = None, temp_adjust_factor=None, few_shot_no = 3, section="train", current_time = None, print_output = False):

    #model_name = "gpt-4o-mini-2024-07-18"
    #model_name = "chatgpt-4o-latest"
    #model_name = "gpt-4o-2024-08-06"
    #model_name = "gpt-3.5-turbo-1106"
    if dataset_name == "personachat":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text = construct_prompt(dataset, conv_id, prompt_type, few_shot_no, section=section, print_output=print_output)
    elif dataset_name == "movie":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text= construct_prompt_movie(dataset, conv_id, prompt_type, print_output=print_output)

    temperature = get_temperature(temp_adjust_factor, conv_id)
    print(f"Temperature: {temperature}")
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=0.9,
        logprobs=True,
        top_logprobs=5
    )

    generated_response = response.choices[0].message.content

    log_probs = [token_obj.logprob for token_obj in response.choices[0].logprobs.content]

    tokens_list = [token_obj.token for token_obj in response.choices[0].logprobs.content]

    formated_response_to_json = {
        "conv_id": conv_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generated_response": generated_response,
        "target_response": target_response,
        "persona_text": persona_text,
        "log_probs": log_probs,
        "tokens_list": tokens_list
    }

    raw_response_to_json = {
        "raw_response": response.to_dict()
    }
    
    saved_filename = f"{prompt_type}_{model_name}_{current_time}"

    if temp_adjust_factor is not None:
        saved_filename += f"_temperature_change_factor{temp_adjust_factor}"
    
    if dataset_name == "movie":
        saved_filename = f"{dataset_name}_{saved_filename}"
        
    save_prompt_as_array(formated_response_to_json, saved_filename)
    save_prompt_as_array(raw_response_to_json, f"Raw_Response/{dataset_name}_{saved_filename}")

    calculate_metrics(prompt_type, conv_id, generated_response, target_response, user_prompt, persona_text, log_probs, tokens_list, model_name, current_time)


def prompt_azure(model_name, conv_id, prompt_type = "context_only", dataset_name = "movie", dataset = None, few_shot_no = 3, section="train", current_time = None):


    if dataset_name == "personachat":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text = construct_prompt(dataset, conv_id, prompt_type, few_shot_no, section=section)
    elif dataset_name == "movie":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text= construct_prompt_movie(dataset, conv_id, prompt_type)

    response = client_azure.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ],
        temperature=0.9,
        top_p=0.9,
        model_extras={
            "logprobs": True,
            "top_logprobs": 5
        }
    )

    generated_response = response.choices[0].message.content
    formated_response_to_json = {
        "conv_id": conv_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generated_response": generated_response,
        "target_response": target_response,
        "persona_text": persona_text
    }

    

    save_prompt_as_array(formated_response_to_json,f"{prompt_type}_{model_name}_{current_time}")

    calculate_metrics(prompt_type, conv_id, generated_response, target_response, user_prompt, persona_text, model_name=model_name, current_time=current_time)


def prompt_huggingface(model_name, conv_id, prompt_type = "context_only", dataset_name = "movie", dataset = None, few_shot_no = 3, section="train", current_time = None):
    if dataset_name == "personachat":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text = construct_prompt(dataset, conv_id, prompt_type, few_shot_no, section=section)
    elif dataset_name == "movie":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text= construct_prompt_movie(dataset, conv_id, prompt_type)

    client = InferenceClient(
        model = model_name,
        token= config['hf_key']
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        logprobs=True
    )
    generated_response = response.choices[0].message.content

    log_probs = [token_obj.logprob for token_obj in response.choices[0].logprobs.content]

    tokens_list = [token_obj.token for token_obj in response.choices[0].logprobs.content]

    formated_response_to_json = {
        "conv_id": conv_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generated_response": generated_response,
        "target_response": target_response,
        "persona_text": persona_text,
        "log_probs": log_probs,
        "tokens_list": tokens_list
    }
    
    model_name = model_name.split("/")[-1]

    save_prompt_as_array(formated_response_to_json,f"{prompt_type}_{model_name}_{current_time}")

    calculate_metrics(prompt_type, conv_id, generated_response, target_response, user_prompt, persona_text, log_probs, tokens_list, model_name, current_time)



def prompt_ollama(model_name, conv_id, prompt_type = "context_only", dataset_name = "movie", dataset = None, few_shot_no = 3, section="train", current_time = None):
    client_ollama = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # required but ignored
    )
    if dataset_name == "personachat":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text = construct_prompt(dataset, conv_id, prompt_type, few_shot_no, section=section)
    elif dataset_name == "movie":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text= construct_prompt_movie(dataset, conv_id, prompt_type)

    response = client_ollama.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9,
        top_p=0.9,
        model='qwen2:0.5b',
    )
    generated_response = response.choices[0].message.content
    formated_response_to_json = {
        "conv_id": conv_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generated_response": generated_response,
        "target_response": target_response,
        "persona_text": persona_text
    }
    save_prompt_as_array(formated_response_to_json,f"{prompt_type}_{model_name}_{current_time}")

    calculate_metrics(prompt_type, conv_id, generated_response, target_response, user_prompt, persona_text, model_name=model_name, current_time=current_time)



