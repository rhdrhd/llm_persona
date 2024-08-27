from prompt import prompt_chatgpt, prompt_azure, prompt_ollama, prompt_huggingface
from helper import get_filtered_conv_ids, read_json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from convokit import Corpus, download
from datasets import load_dataset
import time
from analyze import plot_avg_metrics

#current_time = time.strftime("%m%d-%H%M")
#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit
#conv_ids = generate_random_conv_ids(total_count=2, range_start=1, range_end=500)
def test(dataset_name, model_name, prompt_type,current_time):
    
    #prompt_type = "task_prompt_context_implicit"
    #model_name = "gpt-4o-mini-2024-07-18"
    if dataset_name == "personachat":
        few_shot_no = 3
        conv_ids = range(0,1000)
        dataset_obj = load_dataset("bavard/personachat_truecased", "full")
    elif dataset_name == "movie":
        few_shot_no = 3
        corpus = Corpus(filename=download("movie-corpus"))
        conv_ids = get_filtered_conv_ids(corpus)
        dataset_obj = corpus

    for idx, conv_id in enumerate(conv_ids):
        print(f"Testing model: {model_name}, Current Progress: {idx/10}%")
        # specific settings are in prompt.py
        if model_name in ["chatgpt-4o-latest", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-3.5-turbo-1106"]:
            prompt_chatgpt(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, temp_adjust_factor=None, few_shot_no = few_shot_no, section="validation", current_time = current_time, print_output = False)
        elif model_name in ["Phi-3-5-mini-instruct"]:
            prompt_azure(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)
        elif model_name in ["qwen0.5b"]:
            prompt_ollama(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)
        elif model_name in ["meta-llama/Meta-Llama-3.1-70B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","Qwen/Qwen2-72B-Instruct","microsoft/Phi-3.5-mini-instruct","microsoft/Phi-3-mini-4k-instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
            prompt_huggingface(model_name, conv_id, prompt_type, dataset_name= dataset_name, dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)


def test_movie_chat(model_name, prompt_type, current_time):

    corpus = Corpus(filename=download("movie-corpus"))
    conv_ids = get_filtered_conv_ids(corpus)
    for idx, conv_id in enumerate(conv_ids):
        print(f"Testing model: {model_name}, Current Progress: {idx+1/10}%")
        prompt_chatgpt(model_name, conv_id, prompt_type, dataset_name="movie",dataset=corpus, print_output=False, current_time=current_time)

def test_multiple_models():
    model_list = ["gpt-4o-mini-2024-07-18"]
    prompt_type = "task_prompt_context_implicit"
    for model in model_list:
        current_time = time.strftime("%m%d-%H%M")

        #test("personachat", prompt_type, current_time)
        test("movie", model, prompt_type, current_time)

#test_persona_chat()
test_multiple_models()
#calculate_metrics_from_json("PersonaChat_Metrics/gpt-4o/experiment1_context_only", "context_only")
#print_avg_metrics("experiment1_metrics")
#selected_metrics = ["BLEU-1", "ROUGE-L", "Distinct-1", "Distinct-2", "Persona Coverage", "Persona F1","Cosine Similarity","Perplexity","Avg Drift Score","Confident Drift 001", "Confident Drift 002","Redefine Cosine Similarity"]
#plot_avg_metrics(["context_only_metrics_Meta-Llama-3.1-8B-Instruct_0827-1204"],selected_metrics=selected_metrics, type="table")
#plot_avg_metrics(["context_only_metrics_chatgpt-4o-latest_0827-0147","context_only_metrics_gpt-4o-2024-08-06_0827-0239", "context_only_metrics_gpt-3.5-turbo-1106_0827-0311","Rebirth\Cornell_Movie_Metrics\movie_context_only_metrics_gpt-4o-mini-2024-07-18_0826-2035"], selected_metrics, "table")


def generate_heat():
    # Get results
    metrics_list = []
    results_list = []
    for id in conv_ids:
        for i in range(0, 5):
            results = prompt_chatgpt(prompt_type, id, few_shot_no)
            prompt_response = results[0]  
            target_response = results[1]
            metrics = calculate_metrics(target_response, prompt_response)
            
            combo_result = dict(conv_id = id, attempt_no = i, generated_response = prompt_response, target_response = target_response, metrics = metrics)
            results_list.append(combo_result)

    df = pd.DataFrame(results_list)
    df['Inter Similarity'] = df['metrics'].apply(lambda x: x['Inter Similarity'])
    # Pivot the DataFrame correctly using keyword arguments
    heatmap_data = df.pivot(index="conv_id", columns="attempt_no", values="Inter Similarity")

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='Blues', linewidths=.5)

    # Add labels and title if necessary
    plt.title('Heatmap of Metrics by Conversation ID and Attempt Number')
    plt.xlabel('Attempt Number')
    plt.ylabel('Conversation ID')

    # Show the plot
    plt.show()

#test_movie_chat()