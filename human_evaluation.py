from preprocess import construct_prompt
from datasets import load_dataset
import os
from openai import OpenAI
import json
from helper import save_prompt_as_array, read_json

with open('config.json', 'r') as file:
    config = json.load(file)

os.environ['OPENAI_API_KEY'] = config['api_key']
client = OpenAI()

def eval_drift():
    conv_ids = range(0,1000)
    #prompt_type = "task_prompt_context_implicit"
    model_name = "chatgpt-4o-latest"
    few_shot_no = 3
    dataset = load_dataset("bavard/personachat_truecased", "full")
    prompt_type = "drift_score_eval"
    for conv_id in conv_ids:
        system_prompt, user_prompt, _, _ = construct_prompt(dataset, conv_id, prompt_type, section="validation")
        response = client.chat.completions.create(
            model = model_name,
            #model="gpt-3.5-turbo-0613", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            top_p=0.2,
            # since persona-chat sets a maximum of 15 words per message
            #max_tokens=100, 
            logprobs=True,
        )
        generated_response = response.choices[0].message.content
        log_probs = [token_obj.logprob for token_obj in response.choices[0].logprobs.content]

        tokens_list = [token_obj.token for token_obj in response.choices[0].logprobs.content]
        formated_response_to_json = {
            "conv_id": conv_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_response": generated_response,
            "log_probs": log_probs,
            "tokens_list": tokens_list
        }
        save_prompt_as_array(formated_response_to_json,f"{prompt_type}")


def eval_drift_prompted_results():
    conv_ids = range(0,1000)
    #prompt_type = "task_prompt_context_implicit"
    model_name = "chatgpt-4o-latest"
    few_shot_no = 3
    dataset = load_dataset("bavard/personachat_truecased", "full")
    data = read_json("Rebirth\PersonaChat_Metrics\gpt-4o-mini-2024-07-18\context_only_gpt-4o-mini-2024-07-18_0822-2155")
    prompt_type = "drift_score_eval_prompted"
    system_prompt = "Considering the dialogue provided, determine whether <speaker1> changes the topic from the previous utterance of <speaker0>. Please output a score between 0 and 1, where 0 indicates no topic change with smooth converstation and 1 indicates an abrupt topic change."
    for conv_id in conv_ids:
        last_query = data[conv_id]['user_prompt'].split("\n")[-2]
        generated_response=  data[conv_id]['generated_response']
        user_prompt = "Dialogue history: \n" + last_query + "\n<speaker1>: " + generated_response
        response = client.chat.completions.create(
            model = model_name,
            #model="gpt-3.5-turbo-0613", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            top_p=0.2,
            # since persona-chat sets a maximum of 15 words per message
            #max_tokens=100, 
            logprobs=True,
        )
        generated_response = response.choices[0].message.content
        log_probs = [token_obj.logprob for token_obj in response.choices[0].logprobs.content]

        tokens_list = [token_obj.token for token_obj in response.choices[0].logprobs.content]
        formated_response_to_json = {
            "conv_id": conv_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_response": generated_response,
            "log_probs": log_probs,
            "tokens_list": tokens_list
        }
        save_prompt_as_array(formated_response_to_json,f"{prompt_type}")
eval_drift_prompted_results()