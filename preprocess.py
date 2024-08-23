import random
from datasets import load_dataset
from transformers import pipeline, logging

logging.set_verbosity_error()

def get_random_conv_id(pool, num_conv):
    random_values = random.sample(pool, num_conv)

    for value in random_values:
        pool.remove(value)
    
    return random_values

def extract_format_data(dataset, conv_id, section='train'):
    dialog = dataset[section].filter(lambda example:example['conv_id']== conv_id)
    persona = dialog['personality'][-1]
    history_convo = dialog['history'][-1]
    usr_response = dialog['candidates'][-1][-1]

    unprocessed = [persona, history_convo, usr_response]

    #preprocess the persona
    persona_processed = "Speaker1 Persona: \n"
    for sen in persona:
        persona_processed += sen + "\n"

    #preprocess the history conversation
    history_convo_processed = "Dialogue history: \n"
    
    #concat all history except the last utter from the bot
    for i in range(0,len(history_convo)-1,2):
        bot_uttr = "<speaker0>: " + history_convo[i]
        user_uttr = "<speaker1>: " + history_convo[i+1]
        full_uttr = bot_uttr + "\n" + user_uttr + "\n"
        history_convo_processed += full_uttr

    #add the last utter from bot
    history_convo_processed += "<speaker0>: " + history_convo[-1] + "\n"

    #preprocess the user response
    usr_response_processed = "<speaker1>: " + usr_response + "\n"

    processed = [persona_processed, history_convo_processed, usr_response_processed]
    return unprocessed, processed

# materials[0] is persona, materials[1] is history conversation processed, materials[2] is user response processed
def create_example(dataset, conv_id, implicit=False):
    _, materials = extract_format_data(dataset, conv_id)
    if implicit:
        return materials[1] + materials[2]
    else:
        return materials[0] + materials[1] + materials[2]
    

def create_few_shot_examples(dataset, conv_id, few_shot_no, section="train",implicit=False):
    max_conv_id = dataset[section]['conv_id'][-1]
    pool = [num for num in range(1, max_conv_id + 1) if num != conv_id]
    random_conv_ids = get_random_conv_id(pool, few_shot_no)

    few_shot_examples = ""
    for index, conv_id in enumerate(random_conv_ids):
        few_shot_examples += f"Demo {index}:\n" + create_example(dataset, conv_id, implicit) + "\n"
    return few_shot_examples

def perform_sentiment_analysis(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
    #sentiment_pipeline = pipeline("sentiment-analysis", model= "cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_result = sentiment_pipeline(text)
    
    return sentiment_result[0]

# updated: bot -> <speaker0>, user -> <speaker1>
def construct_prompt(dataset, conv_id, prompt_type, few_shot_no=1, section= "train", print_output= False):
    #max_conv_id = dataset['train']['conv_id'][-1]
    raw_materials, materials = extract_format_data(dataset, conv_id, section=section)
    raw_target_response = raw_materials[2]
    raw_persona_text = raw_materials[0]

    system_prompt = ""
    user_prompt = ""
    few_shot_examples = ""

    # materials[0] is persona, materials[1] is history conversation processed, materials[2] is user response processed
    # only history dialogue
    if prompt_type == "context_only":
        user_prompt += materials[1] + "<speaker1>:"

    # only history dialogue with small hint
    if prompt_type == "context_hint":
        sentiment_ans = perform_sentiment_analysis(raw_target_response)
        system_prompt += f"Hint: the mood of generated response should be {sentiment_ans['label']} with a score of {sentiment_ans['score']}."
        user_prompt += materials[1] + "<speaker1>:"
        
    # task prompt, history dialogue
    if prompt_type == "task_prompt_context_implicit":
        #version 1 Based on the previous conversation history, generate a response for the user that aligns with their profile and the current context of the discussion.
        #version 2 Considering the user's profile and the context as established in the previous dialogue history, craft a response that is coherent, relevant, and tailored to the <speaker1>'s interests and style of communication.
        system_prompt += "Considering the previous dialogue history, craft a response that is coherent, relevant, and tailored to the interests and style of communication of <speaker1>"
        user_prompt += materials[1] + "<speaker1>:"

    # task prompt, persona, and history dialogue
    if prompt_type == "task_prompt_context_explicit":
        system_prompt += "Given the user's profile as outlined in the provided persona information, and considering the context of the ongoing discussion from the previous dialogue history, craft a response in 15 words that is specifically tailored to resonate with the user's explicit characteristics and maintains the continuity of the dialogue."
        user_prompt += materials[0]+ materials[1] + "<speaker1>:"

    # few-shot demos, history dialogue
    if prompt_type == "few_shot_implicit":
        system_prompt += "Considering the various user profiles and styles depicted in the provided few-shot examples, and the ongoing discussion's context as established in the previous dialogue history, craft a response within 15 words that is coherent and relevant. This response should be adaptable to the general preferences and communication styles observed in the examples, while seamlessly continuing the dialogue."

        #set the implicit flag to True to exclude persona in few-shot demos
        few_shot_examples = create_few_shot_examples(dataset, conv_id, few_shot_no, section=section, implicit=True)
        user_prompt += few_shot_examples + materials[1]+ "<speaker1>:"

    if print_output:
        print("### TASK PROMPT ###\n" + system_prompt)
        print("### USER PROMPT ###\n" + user_prompt)
    

    return system_prompt, user_prompt, raw_target_response, raw_persona_text

def construct_prompt_movie(corpus, conv_id, prompt_type):
    system_prompt = ""
    user_prompt = ""
    persona_text = ""

    convo_df = corpus.get_conversation(conv_id).get_utterances_dataframe()
    reversed_df = convo_df.iloc[::-1]

    # if the utterance number x >= 16, trim it to 16
    # if the utterance number  14 <= x <= 15 utterances, trim it to 14
    # if the utterance number 12 <= x <= 13, trim it to 12
    # this is to ensure the conversation is between 6-8 turns
    if len(reversed_df)>=16:
        trimmed_df = reversed_df.head(16)
    elif len(reversed_df)>=14:
        trimmed_df = reversed_df.head(14)
    else:
        trimmed_df = reversed_df.head(12)

    # get the speaker ids
    speakers = trimmed_df['speaker'].unique()

    # exclude the last utterance from the user
    history_convo = trimmed_df['text'].iloc[:-1].tolist()
    history_convo_processed = "Dialogue history: \n"

    # Concat all history except the last utter from the user
    for count, convo in enumerate(history_convo):
        if reversed_df['speaker'].iloc[count] == speakers[0]:
            bot_uttr = "<speaker0>: " + convo
            history_convo_processed += bot_uttr + "\n"
        else :
            user_uttr = "<speaker1>: " + convo
            history_convo_processed += user_uttr + "\n"

    if trimmed_df['speaker'].iloc[-1] == speakers[0]:
        user_prompt += history_convo_processed + "<speaker0>: "
    else: 
        user_prompt += history_convo_processed + "<speaker1>: "

    # craft different system prompts based on the prompt type
    if prompt_type == "context_only":
        system_prompt = ""
    elif prompt_type == "task_prompt_implicit":
        system_prompt += "Note: Each line starts with a speaker tag indicating who is talking. Considering the speaker's profile and the ongoing discussion's context as established in the previous dialogue history, craft a response that is coherent, relevant, and tailored to the speaker's interests and style of communication."

    # access the last user utterance
    raw_target_response = trimmed_df['text'].iloc[-1]  
    persona_text = raw_target_response 
    return system_prompt, user_prompt, raw_target_response, persona_text
    

#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit
def test():
    dataset = load_dataset("bavard/personachat_truecased", "full")
    result = construct_prompt(dataset, 5, "few_shot_implicit",print_output=True)
    return result
