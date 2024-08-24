import os
from datasets import load_dataset
from openai import OpenAI
from preprocess import construct_prompt, construct_prompt_movie
from analyze import calculate_metrics
import json
from helper import save_prompt_as_array
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

with open('config.json', 'r') as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config['api_key']
client = OpenAI()

os.environ["AZURE_INFERENCE_CREDENTIAL"] = config['azure_key']
client_azure = ChatCompletionsClient(
    endpoint='https://Phi-3-5-mini-instruct-irlfw.eastus.models.ai.azure.com',
    credential=AzureKeyCredential(config['azure_key'])
)



def prompt_chatgpt(model_name, conv_id, prompt_type = "context_only", dataset_name = "movie", dataset = None, few_shot_no = 3, section="train", current_time = None):

    #model_name = "gpt-4o-mini-2024-07-18"
    #model_name = "chatgpt-4o-latest"
    #model_name = "gpt-4o-2024-08-06"
    #model_name = "gpt-3.5-turbo-1106"
    if dataset_name == "personachat":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text = construct_prompt(dataset, conv_id, prompt_type, few_shot_no, section=section)
    elif dataset_name == "movie":
        dataset = dataset
        system_prompt, user_prompt, target_response, persona_text= construct_prompt_movie(dataset, conv_id, prompt_type)

    response = client.chat.completions.create(
        model = model_name,
        #model="gpt-3.5-turbo-0613", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9,
        top_p=0.9,
        # since persona-chat sets a maximum of 15 words per message
        #max_tokens=100, 
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
    

    save_prompt_as_array(formated_response_to_json,f"{prompt_type}_{model_name}_{current_time}")
    save_prompt_as_array(raw_response_to_json, f"Raw_Response/{dataset_name}_{prompt_type}_{model_name}_{current_time}")

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
        max_tokens=256,
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


# need to be fixed, batching is not working with OpenAI API
def prompt_chatgpt_batch(prompt_type, conv_ids, few_shot_no):
    dataset = load_dataset("bavard/personachat_truecased", "full")
    
    system_prompts = []
    user_prompts = []
    target_responses = []

    # Collect prompts for all conversation IDs
    for conv_id in conv_ids:
        system_prompt, user_prompt, target_response = construct_prompt(dataset, conv_id, prompt_type, few_shot_no)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt)
        target_responses.append(target_response)
    
    # Create batch messages for the API request
    messages = [{"role": "system", "content": system_prompt} for system_prompt in system_prompts]
    messages += [{"role": "user", "content": user_prompt} for user_prompt in user_prompts]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.9,
        max_tokens=15,  # since persona-chat sets a maximum of 15 words per message
    )
    
    # Process responses and match to the original prompts
    responses = ["" for _ in conv_ids]
    for i, choice in enumerate(response.choices):
        index = i % len(conv_ids)
        responses[index] = choice.message.content
    
    # Combine responses with user prompts and target responses
    batch_responses = []
    for i in range(len(conv_ids)):
        batch_responses.append({
            "response": responses[i],
            "target_response": target_responses[i],
            "user_prompt": user_prompts[i]
        })
    
    return batch_responses

