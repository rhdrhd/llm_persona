from prompt import prompt_chatgpt
from analyze import calculate_metrics, print_avg_metrics, plot_avg_metrics
from helper import generate_filtered_conv_ids
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from convokit import Corpus, download

#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit
#conv_ids = generate_random_conv_ids(total_count=2, range_start=1, range_end=500)
def test_persona_chat():
    conv_ids = range(0,100)
    prompt_type = "few_shot_implicit"
    few_shot_no = 3
    for id in conv_ids:
        results = prompt_chatgpt(prompt_type, id, few_shot_no, section="validation")
        generated_response = results[0]
        target_response = results[1]
        persona_text = results[3]
        metrics = calculate_metrics(id, target_response, generated_response, persona_text)

    print_avg_metrics("experiment1_metrics")

def test_movie_chat():
    corpus = Corpus(filename=download("movie-corpus"))
    #generate_filtered_conv_ids(corpus, 100)
    filename = 'config.json'
    # Load the updated content from the JSON file
    with open(filename, 'r') as file:
        config_data = json.load(file)
    # Access the list
    conv_ids = config_data['conv_ids']
    print(f"The list loaded from {filename}: {conv_ids}")

    for id in conv_ids:
        results = prompt_chatgpt("task_prompt_context_implicit", id, 0, section="train", corpus=corpus)
        generated_response = results[0]
        target_response = results[1]
        persona_text = results[3]
        metrics = calculate_metrics(id, target_response, generated_response, persona_text)

#test_movie_chat()
#print_avg_metrics("experiment1_metrics")
plot_avg_metrics(["experiment1_metrics", "Cornell_Movie_Metrics/experiment-100testpoints/experiment1_task_prompt_implicit_metrics"])
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