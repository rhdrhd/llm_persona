import random
from datasets import load_dataset

def get_random_conv_id(pool, num_conv):
    random_values = random.sample(pool, num_conv)

    for value in random_values:
        pool.remove(value)
    
    return random_values

def extract_format_data(dataset, conv_id):
    dialog = dataset.filter(lambda example:example['conv_id']== conv_id)

    persona = dialog['train']['personality'][-1]
    history_convo = dialog['train']['history'][-1]
    usr_response = dialog['train']['candidates'][-1][-1]

    #preprocess the persona
    persona_processed = "User Persona: \n"
    for sen in persona:
        persona_processed += sen + "\n"

    #preprocess the history conversation
    history_convo_processed = "Dialogue history: \n"
    
    #concat all history except the last utter from the bot
    for i in range(0,len(history_convo)-1,2):
        bot_uttr = "Bot: " + history_convo[i]
        user_uttr = "User: " + history_convo[i+1]
        full_uttr = bot_uttr + "\n" + user_uttr + "\n"
        history_convo_processed += full_uttr

    #add the last utter from bot
    history_convo_processed += "Bot: " + history_convo[-1] + "\n"

    #preprocess the user response
    usr_response_processed = "User: " + usr_response + "\n"
    return [persona_processed, history_convo_processed, usr_response_processed]

# materials[0] is persona, materials[1] is history conversation, materials[2] is user response
def create_example(dataset, conv_id, implicit=False):
    materials = extract_format_data(dataset, conv_id)
    if implicit:
        return materials[1] + materials[2]
    else:
        return materials[0] + materials[1] + materials[2]
    

def create_few_shot_examples(dataset, conv_id, few_shot_no, implicit=False):
    max_conv_id = dataset['train']['conv_id'][-1]
    pool = [num for num in range(1, max_conv_id + 1) if num != conv_id]
    random_conv_ids = get_random_conv_id(pool, few_shot_no)

    few_shot_examples = ""
    for index, conv_id in enumerate(random_conv_ids):
        few_shot_examples += f"Demo {index}:\n" + create_example(dataset, conv_id, implicit) + "\n"
    return few_shot_examples


def construct_prompt(dataset, conv_id, prompt_type, few_shot_no=1, print_output= False):
    max_conv_id = dataset['train']['conv_id'][-1]
    materials = extract_format_data(dataset, conv_id)
    system_prompt = ""
    user_prompt = ""
    few_shot_examples = ""

    # only history dialogue
    if prompt_type == "context_only":
        user_prompt += materials[1] + "User:"

    # task prompt, history dialogue
    if prompt_type == "task_prompt_context_implicit":
        #version 1 Based on the previous conversation history, generate a response for the user that aligns with their profile and the current context of the discussion.
        system_prompt += "Considering the user's profile and the ongoing discussion's context as established in the previous dialogue history, craft a response that is coherent, relevant, and tailored to the user's interests and style of communication."
        user_prompt += materials[1] + "User:"

    # task prompt, persona, and history dialogue
    if prompt_type == "task_prompt_context_explicit":
        system_prompt += "Given the user's profile as outlined in the provided persona information, and considering the context of the ongoing discussion from the previous dialogue history, craft a response that is specifically tailored to resonate with the user's explicit characteristics and maintains the continuity of the dialogue."
        user_prompt += materials[0]+ materials[1] + "User:"

    # few-shot demos, history dialogue
    if prompt_type == "few_shot_implicit":
        system_prompt += "Considering the various user profiles and styles depicted in the provided few-shot examples, and the ongoing discussion's context as established in the previous dialogue history, synthesize a coherent and relevant response. This response should be adaptable to the general preferences and communication styles observed in the examples, while seamlessly continuing the dialogue."

        #set the implicit flag to True to exclude persona in few-shot demos
        few_shot_examples = create_few_shot_examples(dataset, conv_id, few_shot_no, implicit=True)
        user_prompt += few_shot_examples + materials[1]+ "User:"

    if print_output:
        print("### TASK PROMPT ###\n" + system_prompt)
        print("### USER PROMPT ###\n" + user_prompt)
    
    return system_prompt, user_prompt, materials[2]


#context_only, task_prompt_context_implicit, task_prompt_context_explicit, few_shot_implicit
def test():
    dataset = load_dataset("bavard/personachat_truecased", "full")
    result = construct_prompt(dataset, 1, "few_shot_implicit",print_output=True)
    return result
