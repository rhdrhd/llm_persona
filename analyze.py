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

# generate random conversation IDs
def generate_random_conv_ids(total_count, range_start, range_end):
    return random.sample(range(range_start, range_end + 1), total_count)

#calculate average metrics
def calculate_average_metrics(metrics_list):
    avg_metrics = {}
    metric_keys = metrics_list[0].keys()

    for key in metric_keys:
        avg_metrics[key] = np.mean([metrics[key] for metrics in metrics_list])
    
    return avg_metrics

# implementation of P-Cover
def calculate_p_cover(persona, response):
    return 0

# implementation of BLEU-1, BLEU-2, BLEU-3, and BLEU-4
def calculate_bleu(reference_sentence, candidate_sentence):
    reference = [reference_sentence.split()]
    candidate = candidate_sentence.split()
    
    # Weights for BLEU-1, BLEU-2, BLEU-3, and BLEU-4
    weights = [
        (1.0, 0, 0, 0),        
        (0.5, 0.5, 0, 0),        
        (1.0 / 3, 1.0 / 3, 1.0 / 3, 0), 
        (0.25, 0.25, 0.25, 0.25) 
    ]
    
    # Use smoothing to handle cases with few n-grams
    smoothing_function = SmoothingFunction()
    scores = sentence_bleu(reference, candidate, weights, smoothing_function=smoothing_function.method1)
    return scores

# rouge1, rouge2, rougeL
def calculate_rouge(reference_sentence, candidate_sentence):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_sentence, candidate_sentence)
    return scores

def cosine_similarity_embeddings(sentence1, sentence2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([sentence1, sentence2])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cos_sim

def distinct_1(sentence):
    words = sentence.split()
    unique_unigrams = set(words)
    total_unigrams = len(words)
    if total_unigrams == 0:
        return 0.0
    distinct_1_score = len(unique_unigrams) / total_unigrams
    return distinct_1_score

def distinct_2(sentence):
    words = sentence.split()
    bigrams = list(nltk.bigrams(words))
    unique_bigrams = set(bigrams)
    total_bigrams = len(bigrams)
    if total_bigrams == 0:
        return 0.0
    distinct_2_score = len(unique_bigrams) / total_bigrams
    return distinct_2_score

def calculate_metrics(reference_sentence, candidate_sentence):
    bleu_score = calculate_bleu(reference_sentence, candidate_sentence)
    rouge_scores = calculate_rouge(reference_sentence, candidate_sentence)
    cosine_similarity = cosine_similarity_embeddings(reference_sentence, candidate_sentence)
    distinct_1_score = distinct_1(candidate_sentence)
    distinct_2_score = distinct_2(candidate_sentence)

    return {
        'BLEU-1': bleu_score[0],
        'BLEU-2': bleu_score[1],
        'BLEU-3': bleu_score[2],
        'BLEU-4': bleu_score[3],
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'Cosine Similarity': cosine_similarity,
        'Distinct-1': distinct_1_score,
        'Distinct-2': distinct_2_score
    }

#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit
conv_ids = generate_random_conv_ids(total_count=2, range_start=1, range_end=500)
prompt_type = "task_prompt_context_implicit"
few_shot_no = 0

# Get results
metrics_list = []
results_list = []
for id in conv_ids:
    results = prompt_chatgpt(prompt_type, id, few_shot_no)
    prompt_response = results[0]  
    target_response = results[1][6:]
    metrics = calculate_metrics(target_response, prompt_response)
    metrics_list.append(metrics)
    results_list.append(results)



#metrics_list.append(metrics)

# Calculate average metrics
#average_metrics = calculate_average_metrics(metrics_list)

# Print average metrics
#print("Average Metrics:")
#for metric, value in average_metrics.items():
#    print(f"{metric}: {value:.4f}")

#print(user_prompt)
#response, target_response, user_prompt = prompt_chatgpt("context_only", 1, 0)
#print("Prompt response: " + response)
#print("Target response: " + target_response[6:])

#metrics = calculate_metrics(target_response[6:], response)
#print(metrics)