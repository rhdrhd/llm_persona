import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import numpy as np
import spacy
import math
import gensim.downloader as gensim_downloader
from helper import save_json, read_json
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast, RobertaModel, logging
import torch

logging.set_verbosity_error()
# Load the English NLP model
en_tokenizer = spacy.load('en_core_web_sm')
# Load the GloVe model
glove_model = gensim_downloader.load("glove-twitter-100")

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# generate random conversation IDs
def generate_random_conv_ids(total_count, range_start, range_end):
    return random.sample(range(range_start, range_end + 1), total_count)

def tokenize_with_punctuation(text):
    doc = en_tokenizer(text)
    return [token.text.lower() for token in doc]

def tokenize(text):
    doc = en_tokenizer(text)
    # Filter out punctuation and stop words
    return [token.text.lower() for token in doc if token.is_alpha or token.is_digit]

# Calculate the inverse document frequency (idf) for a given term frequency (tf_j).
def calculate_idf(tf_j):
    return 1 / (1 + math.log(1 + tf_j))

# Calculate the term frequency (tf) using Zipf's Law with Glove-twitter-100 index (idx).
def calculate_tf(idx):

    return 1e6 * (1 / (idx ** 1.07))

def calculate_persona_coverage(response, personas, glove_model):

    # Tokenize the response and persona sentences
    W_Y = set(response.split())
    
    # Initialize the maximum persona coverage score
    max_p_cover = 0
    
    # Iterate over each persona sentence
    for persona in personas:
        W_p_i = set(persona.split())
        
        # Calculate the intersection of words between response and persona
        intersection = W_Y.intersection(W_p_i)
        
        if not intersection:
            continue
        
        # Calculate the sum of idf values for words in the intersection
        sum_idf = 0
        for word in intersection:
            if word in glove_model.key_to_index:
                idx = glove_model.key_to_index[word] + 1  # GloVe index is 0-based, so add 1
                tf = calculate_tf(idx)
                idf = calculate_idf(tf)
                sum_idf += idf
        
        # Normalize by the size of the intersection
        p_cover = sum_idf / len(intersection)
        
        # Update the maximum persona coverage score
        if p_cover > max_p_cover:
            max_p_cover = p_cover
    
    return max_p_cover

def calculate_persona_recall(response, personas):

    W_Y = set(tokenize(response))
    max_recall = 0

    if len(W_Y) == 0:
        return 0
    for persona in personas:
        W_p_i = set(tokenize(persona))
        intersection = W_Y & W_p_i
        recall = len(intersection) / len(W_p_i)
        if recall > max_recall:
            max_recall = recall

    return max_recall

def calculate_persona_precision(response, personas):

    W_Y = set(tokenize(response))
    max_precision = 0
    if len(W_Y) == 0:
        return 0
    for persona in personas:
        W_p_i = set(tokenize(persona))
        intersection = W_Y & W_p_i
        precision = len(intersection) / len(W_Y)
        if precision > max_precision:
            max_precision = precision

    return max_precision

def calculate_p_f1(response, personas):
 
    recall = calculate_persona_recall(response, personas)
    precision = calculate_persona_precision(response, personas)
    if recall + precision == 0:
        return 0
    p_f1 = 2 * recall * precision / (recall + precision)
    return p_f1

# Calculates token and character level overlap ratios.
def calculate_overlap(generated_sen, target_sen):

    # Tokenizing both sentences
    tokens_gen = set(tokenize(generated_sen))
    tokens_target = set(tokenize(target_sen))

    # Character sets
    char_set_gen = set(generated_sen)
    char_set_target = set(target_sen)

    # Token overlap calculations
    token_intersection = tokens_gen.intersection(tokens_target)
    token_union = tokens_gen.union(tokens_target)
    token_overlap_ratio = len(token_intersection) / len(token_union) if token_union else 0

    # Character overlap calculations
    char_intersection = char_set_gen.intersection(char_set_target)
    char_union = char_set_gen.union(char_set_target)
    char_overlap_ratio = len(char_intersection) / len(char_union) if char_union else 0

    return token_overlap_ratio, char_overlap_ratio

# implementation of BLEU-1, BLEU-2, BLEU-3, and BLEU-4
def calculate_bleu(target_sentence, generated_sentence):
    target = [tokenize(target_sentence)]
    generated = tokenize(generated_sentence)
    
    # Weights for BLEU-1, BLEU-2, BLEU-3, and BLEU-4, here is by cumulative weights
    weights = [
        (1.0, 0, 0, 0),        
        (0.5, 0.5, 0, 0),        
        (1.0 / 3, 1.0 / 3, 1.0 / 3, 0), 
        (0.25, 0.25, 0.25, 0.25) 
    ]
    
    # Use smoothing to handle cases with few n-grams
    smoothing_function = SmoothingFunction()
    scores = sentence_bleu(target, generated, weights, smoothing_function=smoothing_function.method1)
    return scores

# rouge1, rouge2, rougeL
def calculate_rouge(target_sentence, generated_sentence):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target_sentence, generated_sentence)
    return scores

def calculate_cosine_similarity_embeddings(sentence1, sentence2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

def calculate_distinct_1(sentence):
    words = tokenize(sentence)
    unique_unigrams = set(words)
    total_unigrams = len(words)
    if total_unigrams == 0:
        return 0.0
    distinct_1_score = len(unique_unigrams) / total_unigrams
    return distinct_1_score

def calculate_distinct_2(sentence):
    words = tokenize(sentence)
    bigrams = list(nltk.bigrams(words))
    unique_bigrams = set(bigrams)
    total_bigrams = len(bigrams)
    if total_bigrams == 0:
        return 0.0
    distinct_2_score = len(unique_bigrams) / total_bigrams
    return distinct_2_score

def calculate_perplexity(log_probs):

    avg_log_prob = sum(log_probs) / len(log_probs)
    
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity



# Function to replace all known misrepresentations
def fix_encoding_issues(text):
    replacement_map = {
        "Â°": "°",
        "âĢ": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "—",
        "Ķ": "—",
        "\u2014": "—",
        # Add more replacements as needed
    }
    for incorrect, correct in replacement_map.items():
        text = text.replace(incorrect, correct)
    return text

def clean_tokens(tokens):
    # List of substrings and special tokens to remove
    substrings_to_remove = ["Ġ", "Ļ", "<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    
    cleaned_tokens = []
    for token in tokens:
        #print("token ", token)
        cleaned_token = token
        for substring in substrings_to_remove:
            cleaned_token = cleaned_token.replace(substring, "")
            cleaned_token = cleaned_token.replace("’", "'")
            cleaned_token = fix_encoding_issues(cleaned_token)
        # Append the token only if it's not empty after cleaning
        if cleaned_token:  # This checks if the token is not an empty string
            cleaned_tokens.append(cleaned_token)
    
    return cleaned_tokens

def calculate_aligned_embedding(gpt_tokens):
    # gpt token separate <speaker1> as <, speaker, 1, >:
    if "".join(gpt_tokens[:4])== "<speaker1>":
        gpt_tokens = gpt_tokens[4:]
    
    gpt_tokens = clean_tokens(gpt_tokens)

    # List of tokens from GPT
    #gpt_tokens = [token['token'] for token in data[2]['raw_response']['choices'][0]['logprobs']['content']]

    # Combine tokens back into a sentence
    sentence = "".join(gpt_tokens)

    # Tokenize using RoBERTa
    roberta_inputs = tokenizer(sentence, return_tensors='pt')

    # Get the contextualized embeddings
    with torch.no_grad():
        outputs = model(**roberta_inputs)
        roberta_embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

    # Convert RoBERTa token IDs back to tokens for alignment
    o_roberta_tokens = tokenizer.convert_ids_to_tokens(roberta_inputs['input_ids'][0][1:])
    roberta_tokens = clean_tokens(o_roberta_tokens)
    # Now, align the GPT tokens with RoBERTa tokens and extract embeddings
    aligned_embeddings = []
    gpt_index = 0  # Track the index of GPT tokens
    roberta_index = 0  # Track the index of RoBERTa tokens
    mismatch_count = 0
    while gpt_index < len(gpt_tokens) and roberta_index < len(roberta_tokens):
        # Strip leading spaces from the current GPT token for comparison
        gpt_token = gpt_tokens[gpt_index]
        stripped_gpt_token = gpt_token.lstrip()

        # Start with the current RoBERTa token (strip "Ġ")
        current_roberta_token = roberta_tokens[roberta_index]

        embedding_sum = roberta_embeddings[0, roberta_index].numpy()
        token_count = 1

        #print("prev ",current_roberta_token)
        #print("current embedding ", embedding_sum[0])
        #print("roberta_index ", roberta_index)
        #print("gpt_token ", stripped_gpt_token)


        # Now handle merging RoBERTa tokens to match the final GPT token
        while len(current_roberta_token) < len(stripped_gpt_token) and roberta_index + 1 < len(roberta_tokens):
            roberta_index += 1
            next_roberta_token = roberta_tokens[roberta_index]
            current_roberta_token += next_roberta_token

            # Sum the embeddings and count the tokens
            embedding_sum += roberta_embeddings[0, roberta_index].numpy()
            token_count += 1

        # Handle case where a single RoBERTa token corresponds to multiple GPT tokens
        while len(stripped_gpt_token) < len(current_roberta_token) and gpt_index + 1 < len(gpt_tokens):
            gpt_index += 1
            next_gpt_token = gpt_tokens[gpt_index].lstrip()
            stripped_gpt_token += next_gpt_token

        # Now handle the case where both GPT and RoBERTa tokens need to be merged
        while len(current_roberta_token) != len(stripped_gpt_token):
            if len(current_roberta_token) < len(stripped_gpt_token) and roberta_index + 1 < len(roberta_tokens):
                roberta_index += 1
                next_roberta_token = roberta_tokens[roberta_index]
                current_roberta_token += next_roberta_token
                embedding_sum += roberta_embeddings[0, roberta_index].numpy()
                token_count += 1
                #print(f"Further merging RoBERTa token '{next_roberta_token}' to form '{current_roberta_token}'")
            elif len(current_roberta_token) > len(stripped_gpt_token) and gpt_index + 1 < len(gpt_tokens):
                gpt_index += 1
                next_gpt_token = gpt_tokens[gpt_index].lstrip()
                stripped_gpt_token += next_gpt_token
                #print(f"Further merging GPT token '{next_gpt_token}' to form '{stripped_gpt_token}'")
            else:
                break

        # Now the tokens should match, so we can average and append the embedding
        if current_roberta_token == stripped_gpt_token:
            averaged_embedding = embedding_sum / token_count
            aligned_embeddings.append(averaged_embedding)
            #print(f"Aligned '{stripped_gpt_token}' to '{current_roberta_token}', averaged embedding shape: {averaged_embedding.shape}")
        else:
            print(f"Warning: Mismatch between GPT token '{stripped_gpt_token}' and RoBERTa token '{current_roberta_token}'")
            mismatch_count += 1
            


        # Move to the next tokens
        gpt_index += 1
        roberta_index += 1

        #print("mismatch count ", mismatch_count)
    return aligned_embeddings

    

# version 0.0.1 cosine_similarity_without_perplexity
def calculate_drift_willingness(user_prompt, prompt_type):
    # extract content except last label
    history = user_prompt.split('\n')[:-1]
    start_index = history.index("Dialogue history: ")
    # further extract and remove all irrelevant content in the beginning
    dialogue_lines = history[start_index+1:]

    # remove the Bot and User labels
    dialogue_list_no_labels = [line.split(": ", 1)[1] for line in dialogue_lines]
    history_convo = dialogue_list_no_labels
    cos_sim_list = []
    count = 0
    firstspeaker_name = dialogue_lines[0].split(": ", 1)[0]

    #default calculating second speaker's willingness to drift
    for i in range(0,len(history_convo)-1,2):
        firstspeaker_uttr = history_convo[i]
        secondspeaker_uttr =  history_convo[i+1]
        cos_sim = calculate_cosine_similarity_embeddings(firstspeaker_uttr, secondspeaker_uttr)
        cos_sim_list.append(cos_sim)
        #print(f"Pair {count}:")
        #print(firstspeaker_uttr)
        #print(secondspeaker_uttr)
        count += 1

    avg_willingness = sum(cos_sim_list)/len(cos_sim_list)
    return avg_willingness

def get_token_embedding(sentence, model_name):
    sentence = sentence.lower()

    # Tokenize input
    inputs = tokenizer(sentence, return_tensors='pt')
    # Generate contextualized embeddings
    with torch.inference_mode():
        outputs = model(**inputs)
    # The last_hidden_state contains the embeddings
    token_embedding = outputs.last_hidden_state

    # Convert token IDs to word tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return token_embedding, tokens

# version 0.0.2 cosine_similarity_with_perplexity
def calculate_drift_perplexity(user_prompt, log_probs, gpt_tokens):
    aligned_embedding = calculate_aligned_embedding(gpt_tokens)
    # remove Dialogue History: and User: 
    # extract content except last label
    history = user_prompt.split('\n')[:-1]
    start_index = history.index("Dialogue history: ")
    # further extract and remove all irrelevant content in the beginning
    dialogue_lines = history[start_index+1:]

    # remove the Bot and User labels
    dialogue_list_no_labels = [line.split(": ", 1)[1] for line in dialogue_lines]
    # Get the last utterance
    last_utterance = dialogue_list_no_labels[-1]

    # Compute the sentence embedding by averaging all token embeddings
    last_utterance_embedding, last_utterance_t= get_token_embedding(last_utterance, model_name='roberta-base')
    sentence_embedding = torch.mean(last_utterance_embedding, dim=1)
    
    confident_perplexity_001 = 0.0
    confident_perplexity_002 = 0.0
    redefined_cos_sim = 0.0
    idx = 0
    for idx, token_embedding in enumerate(aligned_embedding):
        token_embedding = torch.tensor(token_embedding).reshape(1, -1)
        cos_sim = cosine_similarity(sentence_embedding, token_embedding)
        confident_perplexity_001 += np.log(cos_sim) + log_probs[idx]
        confident_perplexity_002 += cos_sim * log_probs[idx]
        redefined_cos_sim += cosine_similarity(sentence_embedding, token_embedding*log_probs[idx])

    confident_perplexity_001 = np.exp(-confident_perplexity_001/(idx+1))
    confident_perplexity_002 = np.exp(-confident_perplexity_002/(idx+1))
    redefined_cos_sim = redefined_cos_sim/(idx+1)

    return confident_perplexity_001, confident_perplexity_002, redefined_cos_sim
        


def calculate_metrics(prompt_type, id, generated_sentence, target_sentence, user_prompt, persona, log_probs=None, tokens_list=None, model_name=None, current_time=None):
    
    bleu_score = calculate_bleu(target_sentence, generated_sentence)
    rouge_scores = calculate_rouge(target_sentence, generated_sentence)
    cosine_similarity = calculate_cosine_similarity_embeddings(target_sentence, generated_sentence)
    distinct_1_score = calculate_distinct_1 (generated_sentence)
    distinct_2_score = calculate_distinct_2 (generated_sentence)
    token_overlap_ratio, char_overlap_ratio = calculate_overlap(target_sentence, generated_sentence)
    inter_similarity = sum([bleu_score[0], rouge_scores['rougeL'].fmeasure, cosine_similarity,token_overlap_ratio, char_overlap_ratio])/5

    persona_coverage = calculate_persona_coverage(generated_sentence, persona, glove_model)
    persona_recall = calculate_persona_recall(generated_sentence, persona)
    persona_precision = calculate_persona_precision(generated_sentence, persona)
    p_f1 = calculate_p_f1(generated_sentence, persona)
    perplexity = calculate_perplexity(log_probs)
    drift_willingness = calculate_drift_willingness(user_prompt, prompt_type)
    drift_willingness_new = calculate_drift_perplexity(user_prompt, log_probs, tokens_list)
    metrics = {
        'Conversation ID': id,
        'BLEU-1': bleu_score[0],
        'BLEU-2': bleu_score[1],
        'BLEU-3': bleu_score[2],
        'BLEU-4': bleu_score[3],
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'Cosine Similarity': cosine_similarity,
        'Distinct-1': distinct_1_score,
        'Distinct-2': distinct_2_score,
        'Token Overlap Ratio': token_overlap_ratio,
        'Character Overlap Ratio': char_overlap_ratio,
        'Inter Similarity': inter_similarity,
        'Persona Coverage': persona_coverage,
        'Persona Recall': persona_recall,
        'Persona Precision': persona_precision,
        'Persona F1': p_f1,
        'Perplexity': perplexity,
        'Avg Drift Score': drift_willingness,
        'Confident Drift 001': drift_willingness_new[0],
        'Confident Drift 002': drift_willingness_new[1],
        'Redefine Cosine Similarity': drift_willingness_new[2]
    }
    save_json(metrics, f"{prompt_type}_metrics_{model_name}_{current_time}")

def calculate_avg_metrics(data, selected_metrics=None):
    if selected_metrics is None: 
        avg_metrics = {
            'BLEU-1': 0,
            'BLEU-2': 0,
            'BLEU-3': 0,
            'BLEU-4': 0,
            'ROUGE-1': 0,
            'ROUGE-2': 0,
            'ROUGE-L': 0,
            'Cosine Similarity': 0,
            'Distinct-1': 0,
            'Distinct-2': 0,
            'Token Overlap Ratio': 0,
            'Character Overlap Ratio': 0,
            'Inter Similarity': 0,
            'Persona Coverage': 0,
            'Persona Recall': 0,
            'Persona Precision': 0,
            'Persona F1': 0
        }
    else:
        avg_metrics = {metric: 0 for metric in selected_metrics}
        avg_metrics['Conversation ID'] = 0
        
    for object in data:
        for key in avg_metrics.keys():
                avg_metrics[key] += object[key]
    
    num_objects = len(data)
    for key in avg_metrics.keys():
        if key == 'conv_id' or key == 'Conversation ID':
            continue
        else:
            avg_metrics[key] /= num_objects
            avg_metrics[key] = avg_metrics[key]*100
    return avg_metrics

def calculate_metrics_from_json(filename, prompt_type):
    data = read_json(filename)
    for item in data:
        #print(item['user_prompt'])
        calculate_metrics(prompt_type, item['conv_id'], item['target_response'], item['generated_response'], item['persona_text'])

def print_avg_metrics(filename):
    data = read_json(filename)
    avg_metrics = calculate_avg_metrics(data)
    print(avg_metrics)


def plot_avg_metrics(filenames, selected_metrics=None):
    metrics_list = []
    labels = []

    # Read data from each file and calculate averages
    for filename in filenames:
        data = read_json(filename)
        metrics = calculate_avg_metrics(data, selected_metrics)
        metrics_list.append(metrics)
        labels.append(filename.split('/')[1]+" "+filename.split('/')[2])  # Label each dataset by the file name without the extension

    # Get a list of all metric names except the first item which is conv_id
    if selected_metrics is None:
        metric_keys = list(metrics_list[0].keys())[1:]
    else:
        metric_keys = selected_metrics

    # Get a list of all metric names from the first item in the list (assuming all data have the same metrics)
    n_groups = len(metric_keys)
    n_files = len(filenames)

    # Create arrays for the data
    data = np.array([[metrics[key] for metrics in metrics_list] for key in metric_keys])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.8

    # Create a bar for each file
    for i in range(n_files):
        bars = plt.bar(index + i * bar_width, data[:, i], bar_width, alpha=opacity, label=labels[i])

        # Display the value on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.2f}',  # Format the value to 1 decimal places
                ha='center',
                va='bottom',
                fontsize=5,
                rotation=25
            )

    plt.xlabel('Metrics')
    
    plt.ylabel('Values')
    plt.title('Comparison of Metrics Across Files')
    plt.xticks(index + bar_width * (n_files - 1) / 2, metric_keys, rotation=45, ha='right')

    plt.yticks(np.arange(0, 101, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparison.png")
    plt.show()
    

def test():
    personas = [
        "The cat is happy",
        "The dog sat on the log",
        "The cat chased the dog"
    ]
    generated_sen = "A fast brown fox leaps over a sleeping dog."
    target_sen = "The quick brown fox jumps over the lazy dog."

    metrics = calculate_metrics(target_sen, generated_sen, personas)
    save_json(metrics, "experiment1_metrics")
    print(metrics)