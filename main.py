from prompt import prompt_chatgpt, prompt_azure
from helper import generate_filtered_conv_ids, read_json
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
def test_persona_chat(model_name, prompt_type,current_time):
    conv_ids = range(0,10)
    #prompt_type = "task_prompt_context_implicit"
    #model_name = "gpt-4o-mini-2024-07-18"
    few_shot_no = 3
    dataset_obj = load_dataset("bavard/personachat_truecased", "full")
    for id in conv_ids:
        # specific settings are in prompt.py
        prompt_azure(model_name, id, prompt_type, dataset_name= "personachat", dataset = dataset_obj, few_shot_no = few_shot_no, section="validation", current_time = current_time)


def test_movie_chat():
    #generate_filtered_conv_ids(corpus, 100)
    filename = 'conv_ids'
    # Load the updated content from the JSON file
    config_data = read_json(filename)
    corpus = Corpus(filename=download("movie-corpus"))
    # Access the list value 561 is the sweet spot that reaches the max of memory usage
    conv_ids = config_data['conv_ids'][:561]

    prompt_type = "context_only"
    print("the list loaded")
    for id in conv_ids:
        results = prompt_chatgpt(id, prompt_type, dataset_name="movie",dataset=corpus)

def test_multiple_models():
    model_list = ['Phi-3-5-mini-instruct']
    prompt_type = "context_only"
    for model in model_list:
        current_time = time.strftime("%m%d-%H%M")
        test_persona_chat(model, prompt_type,current_time)

#test_movie_chat()
#test_persona_chat()
test_multiple_models()
#calculate_metrics_from_json("PersonaChat_Metrics/gpt-4o/experiment1_context_only", "context_only")
#print_avg_metrics("experiment1_metrics")
#selected_metrics = ["BLEU-1", "ROUGE-L", "Distinct-1", "Distinct-2", "Persona Coverage", "Persona F1","Cosine Similarity","Perplexity","Avg Drift Score","Confident Drift 001", "Confident Drift 002","Redefine Cosine Similarity"]
#plot_avg_metrics(["Rebirth/PersonaChat_Metrics/chatgpt-4o-latest/task_prompt_context_explicit_metrics_chatgpt-4o-latest_0823-1259", "Rebirth/PersonaChat_Metrics/gpt-4o-2024-08-06/task_prompt_context_explicit_metrics_gpt-4o-2024-08-06_0823-1321","Rebirth/PersonaChat_Metrics/gpt-4o-mini-2024-07-18/task_prompt_context_explicit_metrics_gpt-4o-mini-2024-07-18_0823-1252","Rebirth/PersonaChat_Metrics/gpt-3.5-turbo-1106/task_prompt_context_explicit_metrics_gpt-3.5-turbo-1106_0823-1206"], selected_metrics, "table")

#corpus = Corpus(filename=download("movie-corpus"))
#generate_filtered_conv_ids(corpus, 1000)


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