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
5. Run main.py if all the dependencies required for the current project are already installed. **In main.py file, multi model inference is by default set as False, and test datapoints are set as 20 by default**
```
python main.py
```
## Notes
The previous prompts and related performance metrics obtained from LLMs are stored in the data folder. 
Thesis related images are in image folder.

