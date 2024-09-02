import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from convokit import Corpus, download
from datasets import load_dataset
from helper import get_filtered_conv_ids
from analyze import plot_avg_metrics
from prompt import prompt_chatgpt, prompt_azure, prompt_ollama, prompt_huggingface

#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit

with open('model_selection_list.json', 'r') as file:
    model_config = json.load(file)

with open('metric_selection_list.json', 'r') as file:
    metric_config = json.load(file)

def test_single_model(dataset_name, model_name, prompt_type, sample_no, current_time):

    if dataset_name == "personachat":
        few_shot_no = 3
        #set range as 0 to 1000 to test all the conversations
        conv_ids = range(0, sample_no)
        dataset_obj = load_dataset("bavard/personachat_truecased", "full")
    elif dataset_name == "movie":
        few_shot_no = 3
        corpus = Corpus(filename=download("movie-corpus"))
        conv_ids = get_filtered_conv_ids(corpus)
        conv_ids = conv_ids[:sample_no]
        dataset_obj = corpus
        

    for idx, conv_id in enumerate(conv_ids):
        print(f"Testing model: {model_name}, Current Progress: {(idx+1)*100/len(conv_ids)}%")
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

def test_multiple_models(dataset_name, model_list, prompt_type, sample_no,current_time):
    for model in model_list:
        test_single_model(dataset_name, model, prompt_type, sample_no, current_time)


def __main__():

    # switch to False to test single model
    multi_mode = True

    prompt_type = "context_only"
    dataset_name = "personachat"
    comparison_type = "compare_with_baselines"
    sample_no = 20
    current_time = time.strftime("%m%d-%H%M")

    if  multi_mode == False:
        # choose from model_list
        model_name = "gpt-4o-mini-2024-07-18"
        test_single_model(dataset_name, model_name, prompt_type, sample_no, current_time)
        plot_avg_metrics([f"{prompt_type}_metrics_{model_name}_{current_time}"],selected_metrics= metric_config['compare_with_baselines'], type="table")

    else:
        # support model_list: model_openai, model_azure, model_ollama, model_hf
        model_list = model_config['model_openai'] 
        test_multiple_models(dataset_name, model_list, prompt_type, sample_no,  current_time)
        filename_list = [f"{prompt_type}_metrics_{model}_{current_time}" for model in model_list]
        plot_avg_metrics(filename_list, selected_metrics= metric_config[comparison_type], type="table")
    
if __name__ == "__main__":
    __main__()

