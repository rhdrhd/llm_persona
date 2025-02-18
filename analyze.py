import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import numpy as np
import pandas as pd
import spacy
import math
import gensim.downloader as gensim_downloader
from helper import save_json, read_json
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast, RobertaModel, logging, AutoTokenizer, AutoModelForSequenceClassification
import torch
import concurrent.futures
from functools import lru_cache
import os
import seaborn as sns
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.set_verbosity_error()

# Initialize variables to None
nli_tokenizer = None
nli_model = None
en_tokenizer = None
glove_model = None
roberta_tokenizer = None
roberta_model = None

# load these models if needed, and if loaded already, do not load again
def load_nli_model(use_gpu=True, use_fp16=True):
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None:
        nli_model_path = "zayn1111/deberta-v3-dnli"
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path, use_fast=False, model_max_length=512)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)


def load_spacy_model():
    global en_tokenizer
    if en_tokenizer is None:
        en_tokenizer = spacy.load('en_core_web_sm')

def load_glove_model():
    global glove_model
    if glove_model is None:
        glove_model = gensim_downloader.load("glove-twitter-100")

def load_roberta_model():
    global roberta_tokenizer, roberta_model
    if roberta_tokenizer is None or roberta_model is None:
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        roberta_model = RobertaModel.from_pretrained('roberta-base')

@lru_cache(maxsize=1)
def get_sentence_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# generate random conversation IDs
def generate_random_conv_ids(total_count, range_start, range_end):
    return random.sample(range(range_start, range_end + 1), total_count)

def tokenize_with_punctuation(text):
    doc = en_tokenizer(text)
    return [token.text.lower() for token in doc]

def tokenize(text):
    doc = en_tokenizer(text)
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
        if len(W_p_i) == 0:
            return 0
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
    model = get_sentence_model(model_name)
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

def calculate_c_score(generated_sentence, personas):
    inputs = nli_tokenizer(personas, [generated_sentence] * len(personas), truncation=True, padding=True, return_tensors="pt")
    inputs = {key: val.to(nli_model.device) for key, val in inputs.items()} 
    outputs = nli_model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    label_map = {"entailment": 1, "neutral": 0, "contradiction": -1}
    label_names = ["entailment", "neutral", "contradiction"]
    total = sum(label_map[label_names[torch.argmax(pred)]] for pred in predictions)
    return total

def calculate_coh_con_score(user_prompt, generated_sentence, personas):
    P_list = personas
    Q = user_prompt.split("\n")[-2].split(": ")[1]
    R = generated_sentence
    input_P_R = nli_tokenizer(P_list, [R] * len(P_list), truncation=True, padding=True, return_tensors="pt")
    input_Q_R = nli_tokenizer([Q] * len(P_list), [R] * len(P_list), truncation=True, padding=True, return_tensors="pt")

    # Move inputs to device
    input_P_R = {key: val.to(nli_model.device) for key, val in input_P_R.items()}  
    input_Q_R = {key: val.to(nli_model.device) for key, val in input_Q_R.items()}  

    output_P_R = nli_model(**input_P_R)
    output_Q_R = nli_model(**input_Q_R)

    predictions_P_R = torch.softmax(output_P_R.logits, dim=-1)
    predictions_Q_R = torch.softmax(output_Q_R.logits, dim=-1)

    label_map = {"entailment": 1, "neutral": 0, "contradiction": -1}
    label_names = ["entailment", "neutral", "contradiction"]

    total_coh_con = 0.0
    total_con = 0.0

    for pred1, pred2 in zip(predictions_P_R, predictions_Q_R):
        label_value1 = label_map[label_names[torch.argmax(pred1)]]
        label_value2 = label_map[label_names[torch.argmax(pred2)]]

        if label_value1 + label_value2 == 2:
            total_coh_con += 1
            total_con += 1
        elif label_value1 == 1:
            total_con += 1
    
    total_coh_con /= len(P_list)
    total_con /= len(P_list)
    return total_con, total_coh_con

def calculate_avg_coherence(filename, batch_size=8):
    load_nli_model()

    # Ensure model and tokenizer are on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nli_model.to(device)

    data = read_json(filename)
    
    def process_batch(batch):
        batch_c_scores = []
        batch_con_scores = []
        batch_coh_con_scores = []
        for item in batch:
            c_score = calculate_c_score(item["generated_response"], item["persona_text"])
            con_score, coh_con_score = calculate_coh_con_score(item["user_prompt"], item["generated_response"], item["persona_text"])
            batch_c_scores.append(c_score)
            batch_con_scores.append(con_score)
            batch_coh_con_scores.append(coh_con_score)
        return batch_c_scores, batch_con_scores, batch_coh_con_scores

    # Initialize totals
    total_c = 0.0
    total_con = 0.0
    total_coh_con = 0.0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_c_scores, batch_con_scores, batch_coh_con_scores = process_batch(batch)
        total_c += sum(batch_c_scores)
        total_con += sum(batch_con_scores)
        total_coh_con += sum(batch_coh_con_scores)

    # Calculate averages
    avg_c_score = total_c / len(data)
    avg_con_score = total_con / len(data)
    avg_coh_con_score = total_coh_con / len(data)
    
    return avg_c_score, avg_con_score, avg_coh_con_score


def calculate_perplexity(log_probs):

    avg_log_prob = sum(log_probs) / len(log_probs)
    
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity

# Function to replace all known misrepresentations
def fix_encoding_issues(text):
    replacement_map = {
        "Â°": "°",
        "'Ãº'": "ú",
        "âĢĶ": '—',
        "âĢ": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "—",
        "Ķ": "—",
        "\u2014": "—",
        "Ċ": "\n"
    }
    for incorrect, correct in replacement_map.items():
        text = text.replace(incorrect, correct)
    return text

def clean_tokens(tokens):
    # List of substrings and special tokens to remove
    substrings_to_remove = ["Ċ","\n","Ġ", "Ļ", "<|eot_id|>","<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    
    cleaned_tokens = []
    for token in tokens:
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
    # Clean the GPT tokens
    gpt_tokens = clean_tokens(gpt_tokens)

    # Combine tokens back into a sentence
    sentence = "".join(gpt_tokens)
    # Tokenize using RoBERTa
    roberta_inputs = roberta_tokenizer(sentence, return_tensors='pt')

    # Get the contextualized embeddings
    with torch.no_grad():
        outputs = roberta_model(**roberta_inputs)
        roberta_embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

    # Convert RoBERTa token IDs back to tokens for alignment
    o_roberta_tokens = roberta_tokenizer.convert_ids_to_tokens(roberta_inputs['input_ids'][0][1:])
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

            elif len(current_roberta_token) > len(stripped_gpt_token) and gpt_index + 1 < len(gpt_tokens):
                gpt_index += 1
                next_gpt_token = gpt_tokens[gpt_index].lstrip()
                stripped_gpt_token += next_gpt_token
            else:
                break

        # Now the tokens should match, so we can average and append the embedding
        if current_roberta_token == stripped_gpt_token:
            averaged_embedding = embedding_sum / token_count
            aligned_embeddings.append(averaged_embedding)

        else:
            mismatch_count += 1

        # Move to the next tokens
        gpt_index += 1
        roberta_index += 1

    return aligned_embeddings

    

# version 0.0.1 cosine_similarity_without_perplexity
def calculate_drift_willingness(user_prompt, prompt_type):
    if prompt_type in ["query_only", "context_only_wo_label", "few_shot_implicit","crazy_random_context"] or user_prompt is None:
        return 0.0
    # extract content except last label
    history = user_prompt.split('\n')[:-1]
    start_index = history.index("Dialogue history: ")
    # further extract and remove all irrelevant content in the beginning
    dialogue_lines = history[start_index+1:]

    # remove the Bot and User labels
    dialogue_list_no_labels = [line.split(": ", 1)[1] for line in dialogue_lines]
    history_convo = dialogue_list_no_labels
    cos_sim_list_second_speaker = []
    cos_sim_list_first_speaker = []
    count = 0

    for i in range(0,len(history_convo)-1,2):
        firstspeaker_uttr = history_convo[i]
        secondspeaker_uttr =  history_convo[i+1]
        sec_firstspeaker_uttr = history_convo[i+2]
        cos_sim = calculate_cosine_similarity_embeddings(secondspeaker_uttr, sec_firstspeaker_uttr)
        cos_sim_list_first_speaker.append(cos_sim)

        cos_sim = calculate_cosine_similarity_embeddings(firstspeaker_uttr, secondspeaker_uttr)
        cos_sim_list_second_speaker.append(cos_sim)
        count += 1
    sd_willingess = np.std(cos_sim_list_second_speaker, ddof=1)
    avg_willingness_second = sum(cos_sim_list_second_speaker)/len(cos_sim_list_second_speaker)
    avg_willingness_first = sum(cos_sim_list_first_speaker)/len(cos_sim_list_first_speaker)
    return avg_willingness_second, avg_willingness_first, sd_willingess

def get_token_embedding(sentence):
    sentence = sentence.lower()

    # Tokenize input
    inputs = roberta_tokenizer(sentence, return_tensors='pt')
    # Generate contextualized embeddings
    with torch.inference_mode():
        outputs = roberta_model(**inputs)
    # The last_hidden_state contains the embeddings
    token_embedding = outputs.last_hidden_state

    # Convert token IDs to word tokens
    tokens = roberta_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return token_embedding, tokens

# version 0.0.2 cosine_similarity_with_perplexity
def calculate_drift_perplexity(prompt_type, user_prompt, log_probs, gpt_tokens):
    if prompt_type in ["few_shot_implicit", "crazy_random_context"] or user_prompt is None:
        return [0, 0, 0]
    aligned_embedding = calculate_aligned_embedding(gpt_tokens)
    # remove Dialogue History: and User: 
    # extract content except last label
    history = user_prompt.split('\n')[:-1]
    start_index = history.index("Dialogue history: ")
    # further extract and remove all irrelevant content in the beginning
    dialogue_lines = history[start_index+1:]

    if prompt_type in ["context_only_wo_label"]:
        dialogue_list_no_labels = dialogue_lines
    else:
        # remove the Bot and User labels
        dialogue_list_no_labels = [line.split(": ", 1)[1] for line in dialogue_lines]

    # Get the last utterance
    last_utterance = dialogue_list_no_labels[-1]

    # Compute the sentence embedding by averaging all token embeddings
    last_utterance_embedding, last_utterance_t= get_token_embedding(last_utterance)
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
        
def calculate_basic_metrics(target_sentence, generated_sentence):
    load_spacy_model()
    return (
        calculate_bleu(target_sentence, generated_sentence),
        calculate_rouge(target_sentence, generated_sentence),
        calculate_cosine_similarity_embeddings(target_sentence, generated_sentence),
        calculate_distinct_1(generated_sentence),
        calculate_distinct_2(generated_sentence),
        calculate_overlap(target_sentence, generated_sentence)
    )

def calculate_persona_metrics(generated_sentence, persona):
    if persona is not None:
        load_glove_model()
        return (
            calculate_persona_coverage(generated_sentence, persona, glove_model),
            calculate_persona_recall(generated_sentence, persona),
            calculate_persona_precision(generated_sentence, persona),
            calculate_p_f1(generated_sentence, persona),
        )
    else:
        return (0, 0, 0, 0)

def calculate_perplexity_metrics(log_probs, prompt_type, user_prompt, tokens_list):
    if log_probs is not None:
        load_roberta_model()
        return(
            calculate_perplexity(log_probs),
            calculate_drift_perplexity(prompt_type, user_prompt, log_probs, tokens_list)
        )
    else:
        return (0, [0, 0, 0])

def calculate_metrics(prompt_type, id, generated_sentence, target_sentence, user_prompt=None, persona=None, log_probs=None, tokens_list=None, model_name=None, current_time=None):
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        basic_metrics_future = executor.submit(calculate_basic_metrics, target_sentence, generated_sentence)
        persona_metrics_future = executor.submit(calculate_persona_metrics, generated_sentence, persona)
        drift_willingness_future = executor.submit(calculate_drift_willingness, user_prompt, prompt_type)
        perplexity_metrics_future = executor.submit(calculate_perplexity_metrics, log_probs, prompt_type, user_prompt, tokens_list)

    bleu_score, rouge_scores, cosine_similarity, distinct_1_score, distinct_2_score, overlap_ratio = basic_metrics_future.result()
    persona_coverage, persona_recall, persona_precision, p_f1 = persona_metrics_future.result()
    drift_willingness = drift_willingness_future.result()
    perplexity, drift_willingness_new = perplexity_metrics_future.result()

    inter_similarity = sum([bleu_score[0], rouge_scores['rougeL'].fmeasure, cosine_similarity,overlap_ratio[0], overlap_ratio[1]])/5

    metrics = {
        'Conversation ID': id,
        'BLEU-1': bleu_score[0],
        'BLEU-2': bleu_score[1],
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'Cosine Similarity': cosine_similarity,
        'Distinct-1': distinct_1_score,
        'Distinct-2': distinct_2_score,
        'Token Overlap Ratio': overlap_ratio[0],
        'Character Overlap Ratio': overlap_ratio[1],
        'Inter Similarity': inter_similarity,
        'Persona Coverage': persona_coverage,
        'Persona Recall': persona_recall,
        'Persona Precision': persona_precision,
        'Persona F1': p_f1,
        'Perplexity': perplexity,
        'Avg Drift Score': drift_willingness[0],
        'Avg Drift Score First Speaker': drift_willingness[1],
        'Drift Variance':drift_willingness[2],
        'Confident Drift 001': drift_willingness_new[0],
        'Confident Drift 002': drift_willingness_new[1],
        'Redefine Cosine Similarity': drift_willingness_new[2],
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
    
    outlier_count = avg_metrics.copy()

    for object in data:
        for key in avg_metrics.keys():
            value = object[key]
            if value is None or np.isnan(value) or np.isinf(value) or value >1000:
                outlier_count[key] += 1
            else: 
                avg_metrics[key] += object[key]
    
    num_objects = len(data)
    
    for key in avg_metrics.keys():
        avg_metrics[key] /= (num_objects - outlier_count[key])
        if key not in ["Perplexity","Confident Drift 001","Confident Drift 002"]:
            avg_metrics[key] = avg_metrics[key]*100
    return avg_metrics

def calculate_metrics_from_json(filename, prompt_type):
    data = read_json(filename)
    for item in data:
        calculate_metrics(prompt_type=prompt_type, id = item['conv_id'], generated_sentence=item['generated_response'], target_sentence=item['target_response'], user_prompt=item['user_prompt'], persona=item['persona_text'], log_probs = item['log_probs'])

def print_avg_metrics(filename):
    data = read_json(filename)
    avg_metrics = calculate_avg_metrics(data)
    print(avg_metrics)


def plot_avg_metrics(filenames, selected_metrics=None, type = "bar"):
    metrics_list = []
    labels = []

    # Read data from each file and calculate averages
    for filename in filenames:
        data = read_json(filename)
        metrics = calculate_avg_metrics(data, selected_metrics)
        metrics_list.append(metrics)
        labels.append(filename.split('/')[-1])  # Label each dataset by the file name without the extension

    # Get a list of all metric names except the first item which is conv_id
    if selected_metrics is None:
        metric_keys = list(metrics_list[0].keys())[1:]
    else:
        metric_keys = selected_metrics

    original_df = pd.DataFrame(metrics_list, index=labels, columns=metric_keys)
    df = original_df.map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    df.to_csv('model_performance_comparison.csv', index=True) 

    if type == "bar":
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

    elif type == "table":
        # Plot the DataFrame as a table in a matplotlib figure
        fig, ax = plt.subplots(figsize=(16, 4))  
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)  
        table.scale(1.2, 1.2)  

        plt.subplots_adjust(left=0.3, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        plt.title('Comparison of Metrics Across Files')
        plt.savefig('metrics_table.png') 
        plt.show() 


def plot_correlation_heatmap(filename, selected_metrics, dataset_name):
    data = read_json(filename)
    drift_score_data = read_json("New Metrics/human_drift_score_extracted_original")
    drifit_sd = read_json("drift_score_sd")
    human_cos = read_json("New Metrics/common_metrics_on_last_query_and_target_response")

    # Standardizing the '1 - Avg Drift Score' column
    scaler = StandardScaler()
    df_selected = pd.DataFrame(data)[selected_metrics]
    df_drift_score = pd.DataFrame(drift_score_data, columns=['Human Drift Score'])
    df_drift_score['Human Drift Score'] = scaler.fit_transform(df_drift_score['Human Drift Score'].values.reshape(-1, 1))
    df_drift_sd = pd.DataFrame(drifit_sd, columns=['Drift Score SD'])
    df_human_cos = pd.DataFrame(human_cos)[['Cosine Similarity']]

    if dataset_name == "personachat":
        df_selected['Human Drift Score'] = df_drift_score['Human Drift Score']
        df_selected['Drift Score SD'] = df_drift_sd['Drift Score SD']
        df_selected['Human Cos'] = df_human_cos['Cosine Similarity']
    correlation_matrix_no_id = df_selected.corr()

    # Generating the heatmap for the updated correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix_no_id, annot=True, cmap='crest', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap (without Conversation ID)")
    plt.savefig("correaltion_heatmap.png")
    plt.show()
