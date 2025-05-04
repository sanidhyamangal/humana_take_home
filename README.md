# Humana Take Home Assignment
Author: Sanidhya Mangal

## Problem Statement
Develop a chatbot that reads [published](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf) material and can answer all user queries related to the research paper.

### Assumptions

#### Functional Assumptions
- Data is loaded from the local filesystem in `.pdf` or `.docx` format.
- The chatbot retains past conversations and handles follow-up questions.
- Responses are based solely on the provided context with no external internet knowledge.
- The knowledge base can be expanded as needed.
- The solution is designed for English language queries using standard linguistic structures.
- It is a uni-modal system (supports text, tables, etc.).
- If the chatbot cannot fulfill a request, it directs users to online resources.
- All responses include proper source citations as references.

#### Non-Functional Assumptions
- The system maintains robust API connectivity with LLM models and retrieval agents.
- Answers are strictly derived from the provided context.
- Research materials supply all necessary context for answering queries.
- The research content is up-to-date, accurate, and free from ambiguities.
- Relevant content is retrieved automatically without human intervention.
- The selected LLM accurately processes both user queries and retrieved context.
- The system minimizes hallucinations by restricting responses to provided information.
- Performance remains stable over time.

## Implementation Approach
The chatbot employs a Retrieval Augmented Generation (RAG) architecture to ensure accurate and context-specific responses:
- **Vectorization Module:** Converts text from data sources into embedding vectors.
  - The project uses the `all-MiniLM-L6-v2` vectorizer from `sentence-transformer` to embed and locally store the information. In production, online vector stores such as `Pinecone` or `AzureAISearch` can be used.
- **Prompt Design:** Custom system prompts guide the chatbot’s behavior, while task prompts are managed by the orchestrator (e.g., `llama-index`). Refinement prompts may be used if necessary.
- **Agent Configuration:** Defines which LLM (`mistral:7b`) to use and sets the agent type (e.g., ReActive, Contextual, or Simple) to ensure context-based responses.
- **Performance Evaluation:** Uses metrics like faithfulness, relevancy, and correctness to assess performance. Test data from randomly selected questions in the research paper is used.
- **User Interface:** A Streamlit-based UI provides a playground for debugging and testing the chatbot prior to API exposure.

## Evaluation Metrics:
To evaluate chatbot's performance, 5 random questions were cherry picked along with ground truth value. 
- **Faithfulness**: Metric to evaluates whether a response is faithful to the contexts (i.e. whether the response is supported by the contexts or hallucinated.). After extensive testing, chatbot had a score of _100%_ on faithfulness.
- **Relevancy**: Metric to evaluates the relevancy of retrieved contexts and response to a query. This evaluator considers the query string, retrieved contexts, and response string. After extensive testing, chatbot had a score of _100%_ on relevancy score, meaning all the responses passed this test.
- **Correctness**: Metric to evaluates the correctness of a question answering system. This evaluator depends on ground truth answer to be provided, in addition to the query string and response string. It outputs a score between 1 and 5, where 1 is the worst and 5 is the best, along with a reasoning for the score. Post testing we got the mean score of _4.0_ signifying `~80%` of accuracy in responses.

### Other metrics
There are several other metrics which could be used for evaluation of chatbot such as:
- ContextRelevancy: Evaluates relevancy of retrieved contexts for the given query, didn't used already covered in relevancy.
- AnswerRelevancy: Evaluates relevancy of answer or response for the given query, didn't used already covered in correctness.
- Guidelines: To test if given query and response passes the provided guidelines. Assuming there is no strict guidelines to follow for development of this chatbot, therefore decided to omit this one.

> IMO, using evaluation metrics such as response time, etc, are trivial as they are directly dependent on model used, API latency, compute resources or configuration of APIs, therefore decided to skip development of script to capture them, and to capture those metrics we can simply add profiler to capture telemetry data and analyze them. 

## Project Structure
```sh
.
├── app.py # entrypoint for streamlit
├── embeddings # vector index dir 
├── poetry.lock
├── pyproject.toml
├── README.md
├── ruff.toml
├── SlamonetalSCIENCE1987.pdf
├── src # root pkg directory
│   └── humana_take_home
│       ├── __init__.py
│       ├── agents # all agent configs
│       │   └── research.py
│       ├── evaluators # evaluator module
│       │   ├── evaluator.py
│       │   └── utils.py
│       ├── prompts # prompts module to house system prompt
│       │   └── research_paper.py
│       ├── scripts # scripts to run evaluator, vector documents
│       │   ├── run_evaluator.py
│       │   └── vectorize_documents.py
│       ├── timer.py
│       ├── utils.py
│       └── vectorizers # vectorizer module to house all vectors
│           └── pdf.py 
├── test_data.csv
└── test_results.csv
```

## Setup and Running Instructions

### Environment Setup
- Requires Python 3.10 or later.
- Uses Poetry for dependency management. Install Poetry as per the instructions [here](https://python-poetry.org/docs/#installation) and run:
  `poetry install`

#### Environment Variables
You can either set them in `.env` file or manually set them refereing to `.env.tempelate`

### Model and API Setup
- The chatbot uses Ollama to run LLM models. Download and install it from [Ollama](https://ollama.com).
- Download the [`mistral:7b`](https://ollama.com/library/mistral) model with:
  `ollama pull mistral:7b`
- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. Change this using environment variables if needed.

### Vectorization
- Pre-computed vector embeddings are available in the `embeddings` directory.
- To generate new embeddings, run:
  `poetry run python src/humana_take_home/vectorize_documents.py [options]`

Available options:
```sh
usage: vectorize documents into text embeddings [-h] [--input_dir INPUT_DIR] [--input_files [INPUT_FILES ...]] [--vector_path VECTOR_PATH]

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        directory path to index all the documents
  --input_files [INPUT_FILES ...]
                        list of files to index
  --vector_path VECTOR_PATH
                        path to flush vectorized indexes
```

### User Interface
- Launch the Streamlit UI by executing:
  `poetry run streamlit run app.py`
- If the UI does not auto-open in your browser, refer to the local URL provided in the console output.

>There could be `RuntimeError` from torch, please free to ignore it, it's some background pkg error, application will still run irrespective of it.

### Evaluation
- Evaluate chatbot performance with:
  `poetry run python src/humana_take_home/scripts/run_evaluator.py [options]`
- Test data is provided in `test_data.csv`; extend this file as needed.

Available Options:
```sh
usage: evaluate trained chatbot [-h] --path_to_csv_file PATH_TO_CSV_FILE [--use_correctness] [--export_results_path EXPORT_RESULTS_PATH]

options:
  -h, --help            show this help message and exit
  --path_to_csv_file PATH_TO_CSV_FILE
                        path to the csv file containing the evaluation data
  --use_correctness     use correctness evaluator
  --export_results_path EXPORT_RESULTS_PATH
                        path to export the evaluation results
```

## Next Steps
- To improve performance of LLM response we can integrate better models with high parameters, as currently, app uses a `7b` param model.
- Vectorized documents could be stored in cloud.
- Add `re_rank` strategies to improve retrieval performance.
- Use parsers like `LlamaParse` to parse text better, and add new metadata to filter research material from single file.
- Expose chatbot as an API for mobile, web or desktop consumption.
