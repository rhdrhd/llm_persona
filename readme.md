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
├── data
│   ├── PersonaChat_Metrics
│   ├── Cornell_Movie_Metrics
│   ├── Human_Performance
├── environment.yml
├── requirements.txt
├── README.md
```


### Performance of ICL in different settings

#### Performance comparison between different schemes using GPT models
| **Scheme**                          | **Model**              | **BLEU-1** | **ROUGE-L** | **Dist.1** | **Dist.2** | **Cos.Sim** | **PPL** | **P.Cover** | **P.F1** | **C.** | **Con.** | **Coh.Con.** |
|-------------------------------------|------------------------|------------|-------------|------------|------------|-------------|----------|-------------|----------|-------|----------|--------------|
| **Context Only**                    |                        |            |             |            |            |             |          |             |          |       |          |              |
|                                     | chatgpt-4o-latest      | 9.43       | 11.95       | 88.96      | 99.25      | 29.54       | 1.66     | **8.02**    | 13.19    | **0.18** | **4.54** | **2.93**     |
|                                     | gpt-4o-2024-08-06      | 10.38      | 12.73       | 91.03      | 99.70      | **_30.48_** | 1.58     | 7.92        | 14.42    | 0.13  | 3.66     | 1.96         |
|                                     | gpt-4o-mini-2024-07-18 | **_11.34_**| **_13.51_** | **93.54**  | **_99.80_**| 30.12       | **_1.41_**| 7.19        | 14.15    | 0.06  | 2.03     | 0.98         |
|                                     | gpt-3.5-turbo-1106     | 9.24       | 11.13       | 90.77      | 99.29      | 24.37       | 1.66     | 7.73        | **14.65**| 0.09  | 3.30     | 1.89         |
| **Task Prompt + Context**           |                        |            |             |            |            |             |          |             |          |       |          |              |
|                                     | chatgpt-4o-latest      | 6.61       | 8.93        | 83.52      | 99.16      | **28.48**   | 1.90     | 9.06        | 12.66    | **0.42** | 10.08    | 6.71         |
|                                     | gpt-4o-2024-08-06      | 6.84       | 9.11        | 85.41      | 99.34      | 27.85       | 1.94     | 9.09        | 13.04    | 0.34  | 8.12     | 4.76         |
|                                     | gpt-4o-mini-2024-07-18 | **7.85**   | **10.22**   | **87.26**  | **99.44**  | 28.27       | **1.65** | 8.72        | 13.69    | 0.22  | 5.43     | 3.55         |
|                                     | gpt-3.5-turbo-1106     | 6.70       | 8.80        | 82.68      | 98.66      | 25.25       | 1.69     | **9.67**    | **14.22**| 0.53  | **12.37**| **8.26**     |
| **Task Prompt + Context + Persona** |                        |            |             |            |            |             |          |             |          |       |          |              |
|                                     | chatgpt-4o-latest      | 6.14       | 8.51        | 81.21      | 98.96      | **29.16**   | 1.93     | 9.94        | 14.24    | 0.96  | 22.01    | **_14.90_**  |
|                                     | gpt-4o-2024-08-06      | 6.34       | 8.67        | **83.39**  | **99.25**  | 28.77       | 1.99     | 10.21       | 15.26    | 0.95  | 21.72    | 12.18        |
|                                     | gpt-4o-mini-2024-07-18 | **6.64**   | **9.09**    | 83.21      | 99.15      | 28.49       | 1.76     | 9.90        | 15.36    | 0.88  | 19.90    | 11.98        |
|                                     | gpt-3.5-turbo-1106     | 6.12       | 8.22        | 80.54      | 98.54      | 24.07       | **1.72** | **_11.32_** | **18.32**| **_1.46_** | **_32.66_**| 14.48        |
| **Few-Shot (3)**                    |                        |            |             |            |            |             |          |             |          |       |          |              |
|                                     | chatgpt-4o-latest      | 7.82       | 10.31       | 87.04      | 99.30      | 28.48       | 1.86     | 8.72        | 12.73    | 0.21  | 5.28     | 3.49         |
|                                     | gpt-4o-2024-08-06      | **8.51**   | **11.00**   | **89.01**  | **99.54**  | **29.15**   | 1.97     | 8.79        | 13.61    | 0.17  | 4.48     | 2.47         |
|                                     | gpt-4o-mini-2024-07-18 | 8.29       | 10.84       | 88.12      | 99.45      | 28.47       | **1.66** | 8.52        | 14.13    | 0.18  | 4.63     | 3.00         |
|                                     | gpt-3.5-turbo-1106     | 7.08       | 9.21        | 83.69      | 98.77      | 23.78       | 1.92     | **8.83**    | **14.81**| **0.39**| **9.37** | **5.01**     |
| **Human Performance**               |                        | -          | -           | **_96.41_**| 99.77      | -           | -        | 5.37        | **_21.20_**| 0.256| 6.33     | 1.03         |


##### Ablation Study (conducted on gpt-4o-mini-2024-07-18)
| Scheme                 |  Status |
|------------------------|--------|
| Full context w/o speaker tag   | &check; |
| Last utterance with speaker tag  | &check; |
| Last utterance + 1 extra round with speaker tag | &check; |
| Last utterance + 2 extra rounds with speaker tag | &check; |
| Last utterance + 3 extra rounds with speaker tag | &check; |
| Last utterance + 4 extra rounds with speaker tag | &check; |



#### 2. Cornell Movie Dialogues

| Scheme                 | Model                 | Status |
|------------------------|-----------------------|--------|
| Context-only           | chatgpt-4o-latest     |        |
|                        | gpt-4o-2024-08-06     |        |
|                        | gpt-4o-mini-2024-07-18|        |
|                        | gpt-3.5-turbo-1106    |        |
| Task Prompt + Context  | chatgpt-4o-latest     |        |
|                        | gpt-4o-2024-08-06     |        |
|                        | gpt-4o-mini-2024-07-18|        |
|                        | gpt-3.5-turbo-1106    |        |
| Task Prompt + Context + Persona | chatgpt-4o-latest     |        |
|                        | gpt-4o-2024-08-06     |        |
|                        | gpt-4o-mini-2024-07-18|        |
|                        | gpt-3.5-turbo-1106    |        |




