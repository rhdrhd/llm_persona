import os
from datasets import load_dataset
from openai import OpenAI
from preprocess import construct_prompt

os.environ['OPENAI_API_KEY'] = 'sk-proj-zgpS1dH01GeQiQzhICjNT3BlbkFJq8oC1i8atPEckRUxahYK'
client = OpenAI()


def prompt_chatgpt(prompt_type, conv_id, few_shot_no):
    dataset = load_dataset("bavard/personachat_truecased", "full")
    system_prompt, user_prompt, target_response = construct_prompt(dataset, conv_id, prompt_type, few_shot_no)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9,  
        max_tokens=15, # since persona-chat sets a maximum of 15 words per message
    )
    return response.choices[0].message.content, target_response, user_prompt

# need to be fixed, batching is not working
def prompt_chatgpt_batch(prompt_type, conv_ids, few_shot_no):
    dataset = load_dataset("bavard/personachat_truecased", "full")
    
    system_prompts = []
    user_prompts = []
    target_responses = []

    # Collect prompts for all conversation IDs
    for conv_id in conv_ids:
        system_prompt, user_prompt, target_response = construct_prompt(dataset, conv_id, prompt_type, few_shot_no)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt)
        target_responses.append(target_response)
    
    # Create batch messages for the API request
    messages = [{"role": "system", "content": system_prompt} for system_prompt in system_prompts]
    messages += [{"role": "user", "content": user_prompt} for user_prompt in user_prompts]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.9,
        max_tokens=15,  # since persona-chat sets a maximum of 15 words per message
    )
    
    # Process responses and match to the original prompts
    responses = ["" for _ in conv_ids]
    for i, choice in enumerate(response.choices):
        index = i % len(conv_ids)
        responses[index] = choice.message.content
    
    # Combine responses with user prompts and target responses
    batch_responses = []
    for i in range(len(conv_ids)):
        batch_responses.append({
            "response": responses[i],
            "target_response": target_responses[i],
            "user_prompt": user_prompts[i]
        })
    
    return batch_responses

