## This is the repository for the Final Thesis Project of MSc IMLS in UCL

In this project, we intend to explore SOTA LLMs capabilities in personalized dialogue generation using implicit persona modeling, with a particular focus on GPT model series by OpenAI. 

## Repository Structure
The current project structure is shown below
```
├── preprocess.py
├── helper.py
├── prompt.py
├── analyze.py
├── main.py
├── environment.yml
├── requirements.txt
├── model_selection_list.json
├── metric_selection_list.json
├── API_KEYS.json
├── conv_ids.json
├── README.md
├── images
├── data
│   ├── PersonaChat_Metrics
│   ├── Cornell_Movie_Metrics
│   ├── Human_Performance

```

## How to start
1. Create a new conda environment from environment.yml file.
```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate llm-persona
```
3. Install spaCy language model
```
python -m spacy download en
```
4. Insert related API_KEY in API_KEYS.json file.
```
    "openai_key": "",
    "azure_key": "",
    "hf_key": ""
```
5. Run main.py if all the dependencies required for the current project are already installed. 
   **multi model inference is by default set as True, testing models are set as model_openai which includes gpt3.5 and gpt4o model series, sample number is set as 20 for demonstration purpose**
```
python main.py
```

## Configurable Parameters
The main.py file contains the following parameters that can be configured for customized experiment. 

| Param Name                 | Options                | Description            |
|----------------------------|------------------------|------------------------|
| multi_mode                 | True, False                 | If True, multiple models will be compared. |
| prompt_type                | ** Separately Listed Below ** |  |
| dataset_name               | personachat, movie          | select the dataset to be used for the experiment. |
| comparison_type            | compare_with_baselines, different_prompt_schemes| select the metrics to show|
| sample_no         | INT                 | Assign the number of samples to be used in the experiment. Set as 20 for testing |

#### Prompt Types
| Prompt Type                | Description            |
|----------------------------|------------------------|
| context_only             | Only using history text         |
| task_prompt_context_implicit | Using task prompt and history text |
| task_prompt_context_explicit | Using task prompt, history text and speaker persona |
| few_shot_implicit       | Using task prompt, dialogue demos, history text|
| context_only_wo_label   | Only using history text without speaker label |
| query_only             | Only using task prompt |
| random_context        | Using shuffled history text at sentence level |
| crazy_random_context | Using shuffled history text at character level |

#### Model Series
| Model Series               | Models included            |
|----------------------------|------------------------|
| model_openai               | gpt-3.5-turbo-1106, gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, chatgpt-4o-latest |
| model_hf                   | Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct |
| model_azure                | Phi-3-5-mini-instruct |
| model_ollama               | Qwen2-0.5B |


## Supported Models
| Model Name                 | Source                 |
|----------------------------|------------------------|
| gpt-3.5-turbo-1106         | OpenAI                 |
| gpt-4o-mini-2024-07-18     | OpenAI                 |
| gpt-4o-2024-08-06          | OpenAI                 |
| chatgpt-4o-latest          | OpenAI                 |
| Meta-Llama-3.1-70B-Instruct| Hugging Face           |
| Meta-Llama-3.1-8B-Instruct | Hugging Face           |
| Mistral-7B-Instruct-v0.3   | Hugging Face           |
| Phi-3-mini-4k-instruct     | Hugging Face           |
| Phi-3-5-mini-instruct      | Azure AI Studio        |
| Qwen2-0.5B                 | Ollama (Need to be locally hosted)|

## Notes
The previous prompts and related performance metrics obtained from LLMs are stored in the data folder. 
Thesis related images are in image folder.

