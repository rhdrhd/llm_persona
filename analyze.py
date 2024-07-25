import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import numpy as np
from prompt import prompt_chatgpt, prompt_chatgpt_batch
import spacy
import math
import gensim.downloader as gensim_downloader
from helper import save_json, read_json

# Load the English NLP model
tokenizer = spacy.load('en_core_web_sm')
# Load the GloVe model
glove_model = gensim_downloader.load("glove-twitter-100")

# generate random conversation IDs
def generate_random_conv_ids(total_count, range_start, range_end):
    return random.sample(range(range_start, range_end + 1), total_count)

def tokenize(text):
    doc = tokenizer(text)
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

    if len(W_Y) or len(W_p_i) == 0:
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

def cosine_similarity_embeddings(sentence1, sentence2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

def distinct_1(sentence):
    words = tokenize(sentence)
    unique_unigrams = set(words)
    total_unigrams = len(words)
    if total_unigrams == 0:
        return 0.0
    distinct_1_score = len(unique_unigrams) / total_unigrams
    return distinct_1_score

def distinct_2(sentence):
    words = tokenize(sentence)
    bigrams = list(nltk.bigrams(words))
    unique_bigrams = set(bigrams)
    total_bigrams = len(bigrams)
    if total_bigrams == 0:
        return 0.0
    distinct_2_score = len(unique_bigrams) / total_bigrams
    return distinct_2_score

def calculate_metrics(id, target_sentence, generated_sentence, persona):
    bleu_score = calculate_bleu(target_sentence, generated_sentence)
    rouge_scores = calculate_rouge(target_sentence, generated_sentence)
    cosine_similarity = cosine_similarity_embeddings(target_sentence, generated_sentence)
    distinct_1_score = distinct_1 (generated_sentence)
    distinct_2_score = distinct_2 (generated_sentence)
    token_overlap_ratio, char_overlap_ratio = calculate_overlap(target_sentence, generated_sentence)
    inter_similarity = sum([bleu_score[0], rouge_scores['rougeL'].fmeasure, cosine_similarity,token_overlap_ratio, char_overlap_ratio])/5

    persona_coverage = calculate_persona_coverage(generated_sentence, persona, glove_model)
    persona_recall = calculate_persona_recall(generated_sentence, persona)
    persona_precision = calculate_persona_precision(generated_sentence, persona)
    p_f1 = calculate_p_f1(generated_sentence, persona)
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
        'Persona F1': p_f1
    }
    save_json(metrics, "experiment1_metrics")
    return metrics

def calculate_avg_metrics(data):
    avg_metrics = {
        'Conversation ID': 0,
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
    for object in data:
        for key in object.keys():
            if key == 'Conversation ID':
                continue
            else: 
                avg_metrics[key] += object[key]
    
    num_objects = len(data)
    for key in avg_metrics.keys():
        if key == 'Conversation ID':
            continue
        else:
            avg_metrics[key] /= num_objects
            avg_metrics[key] = avg_metrics[key]*100
    return avg_metrics

def print_avg_metrics(filename):
    data = read_json(filename)
    avg_metrics = calculate_avg_metrics(data)
    print(avg_metrics)

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