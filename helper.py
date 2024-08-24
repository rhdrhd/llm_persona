import json
import os 
from convokit import Corpus, download

def save_json(new_object, filename):
    converted_data = {}
    for key, value in new_object.items():
        if key == 'Conversation ID' or key == 'conv_id':
            converted_data[key] = value
        else:
            try:
                # Attempt to convert non 'conv_id' values to float
                converted_data[key] = float(value)
            except ValueError:
                # Print or handle values that cannot be converted to float
                print(f"Warning: The value for key '{key}' is not convertible to float: {value}")
                # Optionally, you can assign the original value or handle differently
                converted_data[key] = value

    file_path = f"{filename}.json"
    
    # Read existing data from the file if it exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new object
    data.append(converted_data)

    # Write the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
def read_json(filename):
    with open(f"{filename}.json", "r") as f:
        data = json.load(f)
    return data

# read the data first then append the new object and overwrite the file with new data

def combine_json_files(file1, file2, new_file):
    data1 = read_json(file1)
    data2 = read_json(file2)
    data1.extend(data2)

    with open(f"{new_file}.json", "w") as f:
        json.dump(data1, f, indent=4)
    
def save_prompt_as_array(new_object, filename):
    file_path = f"{filename}.json"
    
    # Read existing data from the file if it exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new object
    data.append(new_object)

    # Write the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        
def generate_filtered_conv_ids(corpus, total_num):
    num_of_utterances = 0
    num_of_speakers = 0
    conv_ids = []
    conv_id = 0
    count = 0
    # only select conversation with 2 speakers and 12, 14, 16 utterances
    while len(conv_ids) < total_num:
        df = corpus.random_conversation().get_utterances_dataframe()
        num_of_utterances = df.shape[0]
        num_of_speakers = df['speaker'].nunique()
        conv_id = df['conversation_id'].iloc[0]
        #print("here")
        if ((num_of_utterances > 11) and (num_of_speakers == 2) and conv_id not in conv_ids):
            conv_ids.append(conv_id)
        count += 1
        print(count)

    filename = 'config.json'

    # Read the existing JSON file
    with open(filename, 'r') as file:
        config_data = json.load(file)

    # Update the content with the new list
    config_data['conv_ids'] = conv_ids
    with open(filename, 'w') as file:
        json.dump(config_data, file, indent=4)

    print(f"The list has been added to {filename}")


