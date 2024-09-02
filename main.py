import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from convokit import Corpus, download
from datasets import load_dataset
from helper import get_filtered_conv_ids, read_json
from analyze import (plot_avg_metrics, calculate_avg_coherence, calculate_drift_perplexity,
                     calculate_perplexity_metrics, plot_correlation_heatmap)
from prompt import prompt_chatgpt, prompt_azure, prompt_ollama, prompt_huggingface

#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit

with open('model_selection_list.json', 'r') as file:
    model_config = json.load(file)

with open('metric_selection_list.json', 'r') as file:
    metric_config = json.load(file)

def test_single_model(dataset_name, model_name, prompt_type,current_time):

    if dataset_name == "personachat":
        few_shot_no = 3
        conv_ids = range(0,2)
        dataset_obj = load_dataset("bavard/personachat_truecased", "full")
    elif dataset_name == "movie":
        few_shot_no = 3
        corpus = Corpus(filename=download("movie-corpus"))
        conv_ids = get_filtered_conv_ids(corpus)
        dataset_obj = corpus

    for idx, conv_id in enumerate(conv_ids):
        print(f"Testing model: {model_name}, Current Progress: {idx/10}%")
        # specific settings are in prompt.py
        if model_name in model_config["model_openai"]:
            prompt_chatgpt(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, temp_adjust_factor=None, few_shot_no = few_shot_no, section="validation", current_time = current_time, print_output = False)
        elif model_name in model_config["model_azure"]:
            prompt_azure(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)
        elif model_name in model_config["model_ollama"]:
            prompt_ollama(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)
        elif model_name in model_config["model_hf"]:
            prompt_huggingface(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)


def test_movie_chat(model_name, prompt_type, current_time):

    corpus = Corpus(filename=download("movie-corpus"))
    conv_ids = get_filtered_conv_ids(corpus)
    for idx, conv_id in enumerate(conv_ids):
        print(f"Testing model: {model_name}, Current Progress: {idx+1/10}%")
        prompt_chatgpt(model_name, conv_id, prompt_type, dataset_name="movie",dataset=corpus, print_output=False, current_time=current_time)

def test_multiple_models(dataset_name, model_list, prompt_type, current_time):
    for model in model_list:
        test_single_model(dataset_name, model, prompt_type, current_time)


def __main__():

    multi_mode = False
    current_time = time.strftime("%m%d-%H%M")
    prompt_type = "context_only"
    dataset_name = "personachat"

    if  multi_mode == False:
        model_name = "gpt-4o-mini-2024-07-18"
        test_single_model(dataset_name, model_name, prompt_type, current_time)
        plot_avg_metrics([f"{prompt_type}_metrics_{model_name}_{current_time}"],selected_metrics= metric_config['compare_with_baselines'], type="table")

    else:
        model_list = model_config['model_openai'] 
        test_multiple_models(dataset_name, model_list, prompt_type, current_time)

if __name__ == "__main__":
    __main__()

#data_new = []
#for item in data['ablation_test']:
#    new_item = item.replace("\\", "/")
#    data_new.append(new_item)
#plot_avg_metrics(["Rebirth/PersonaChat_Metrics/chatgpt-4o-latest/context_only_metrics_chatgpt-4o-latest_0823-0621","Rebirth/PersonaChat_Metrics/gpt-4o-2024-08-06/context_only_metrics_gpt-4o-2024-08-06_0822-2223","Rebirth/PersonaChat_Metrics/gpt-4o-mini-2024-07-18/context_only_metrics_gpt-4o-mini-2024-07-18_0822-2155", "Rebirth/PersonaChat_Metrics/gpt-3.5-turbo-1106/context_only_metrics_gpt-3.5-turbo-1106_0823-0236"], selected_metrics=selected_metrics, type="table")
 
#plot_correlation_heatmap("Rebirth/PersonaChat_Metrics/gpt-4o-2024-08-06/context_only_metrics_gpt-4o-2024-08-06_0822-2223", selected_metrics, "personachat")





def test_perplexity():
    data = read_json("prompt_content_test")
    prompt_type = "context_only"
    user_prompt = data[0]['user_prompt']
    log_probs = data[0]['log_probs']
    tokens_list = data[0]['tokens_list']

    calculate_perplexity_metrics(log_probs, prompt_type, user_prompt, tokens_list)
