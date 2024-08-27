import orjson
import os 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
        with open(file_path, "rb") as f:
            try:
                data = orjson.loads(f.read())
            except orjson.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new object
    data.append(converted_data)

    # Write the updated data back to the file
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
        ))


def read_json(filename):
    file_path = f"{filename}.json"
    
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Attempt to read the JSON file as text
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace non-standard JSON values like 'Infinity' with 'null'
        content = content.replace('Infinity', 'null')
        
        # Parse the modified content using orjson
        data = orjson.loads(content)
        
        return data
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    except orjson.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return None
    
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

# read the data first then append the new object and overwrite the file with new data

def combine_json_files(file1, file2, new_file):
    data1 = read_json(file1)
    data2 = read_json(file2)
    data1.extend(data2)

    with open(f"{new_file}.json", "wb") as f:
        f.write(orjson.dumps(
            data1,
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
        ))
    
def save_prompt_as_array(new_object, filename):
    file_path = f"{filename}.json"
    
    # Read existing data from the file if it exists
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            try:
                data = orjson.loads(f.read())
            except orjson.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new object
    data.append(new_object)

    # Write the updated data back to the file
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
        ))
        
def get_filtered_conv_ids(corpus):
    if os.path.exists("conv_ids.json"):
        with open("conv_ids.json", "rb") as file:
            config_file = orjson.loads(file.read())
            conv_ids = config_file['conv_ids']
            print("The list has been loaded")
            return conv_ids
    else:
            num_of_utterances = 0
            num_of_speakers = 0
            conv_ids = []
            conv_id = 0
            count = 0
            # only select conversation with 2 speakers and 12, 14, 16 utterances
            print("Now filtering the conversations...")
            while len(conv_ids) < 1000:
                df = corpus.random_conversation().get_utterances_dataframe()
                num_of_utterances = df.shape[0]
                num_of_speakers = df['speaker'].nunique()
                conv_id = df['conversation_id'].iloc[0]
                #print("here")
                if ((num_of_utterances > 11) and (num_of_speakers == 2) and conv_id not in conv_ids):
                    conv_ids.append(conv_id)
                count += 1

            filename = 'conv_ids.json'

            data_to_save = {'conv_ids': conv_ids}

            with open(filename, 'wb') as file:
                file.write(orjson.dumps(
                data_to_save,
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
            ))

            print(f"The list has been added to {filename}")
            return conv_ids



def calculate_temperature_adjustment():
    data = read_json("Rebirth\PersonaChat_Metrics\gpt-4o-mini-2024-07-18\context_only_metrics_gpt-4o-mini-2024-07-18_0822-2155")
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    df['normalized_column'] = scaler.fit_transform(df[['Avg Drift Score']])

    # 2. Convert the normalized data to JSON
    json_data = df['normalized_column'].to_json(orient='records')

    # 3. Save the JSON to a file

    with open('normalized_drift.json', 'wb') as f:
        f.write(orjson.dumps(json_data))

    print("Normalized data has been saved to 'normalized_data.json'")

def get_temperature(temp_adjust_factor, conv_id):
    if temp_adjust_factor is None:
        return 0.9
    else:
        with open('normalized_drift.json', 'rb') as f:
            data = orjson.loads(f.read())
        return 0.9 + data[conv_id]/temp_adjust_factor
    