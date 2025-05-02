# Humana Take Home Assignment
Author: Sanidhya Mangal

## Problem Statment
Develop a chatbot to read [published](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf) material and have capabilities to answer all the user queries related to the research paper.

### Assumptions
#### Functional
- All the data loaded would be from local filesystem and would be either `.pdf` or `.docx`
- Chatbot shall be able to retain past conversation and answer follow up questions.
- Chatbot shall only answer from the provided context, nothing from internet knowlege.
- Chatbot shall be able to expand knowledge base as per requirements.
- Chatbot is designed for english language. Users follow standard lingustic structure over complex jargons.
- Chatbot is uni-modal, i.e., only supports text, tables, etc. 
- If chatbot cannot fulfill any user request, shall direct user to look for online material.
- Chatbot should quote its sources or references for all the user queries.

#### Non-Functional
- Chatbot shall maintain proper API connectivity with the LLM models, reterival agent, etc.
- Chatbot should only answers related to the provided context, shall not include any internet knowledge.
- Provided research material provides all the context to answer all the relevant user query.
- The quality of information/research material is up-to-date, accuracte and free of any ambiguity error.
- Chatbot is able to retrieve relevant chunk of material without any human intervention.
- Selected LLM understand both user queries and retrieved context correctly and synthesize the responses.
- Core assumption is LLM will stick to the retrieved chunk of information, minimizing "halucinations".
- System shall have stable performance over the time.

## Approach
To develop an effective chatbot for a QnA we can develop a RAG based agent to effectively serve user requests without fine-tuning LLMs, one key benefit with this approach is, it helps in reducing hallucination within responses and create a better and streamlined. For a RAG agent we need to develop below mentioned modules:
- Develop a vectorizer module to vecotrize all the textual information from data data scource into embedding vectors.
  - For this project we will use mini-llm-v6 vectorizer to embedd all the information and store it as local embeddings. In an ideal real-world production we can use online vector store such as `Pinecone`, `AzureAISearch`, etc.
- Design or write system prompt, user instructions, or task prompts. This is crucial step for any RAG based agent, since performance of GenAI based LLMs are directly correlated to prompts, or instructions supplied.
  - For this project we will write our own custom system prompt, task prompt would be handled by orchestrator -> `llama-index`, if required we can write refinement prompt to enhance response synthesis.
- Agent Config: In this module we would define all the agent configuration, such as which LLM to use, nature of agent whether it is `ReActive`, `Contextual`, `Simple`, etc.
  - Since this tool focuses on simple QnA reterival, therefore we would create a basic agent with context reterival properties to restrict user to only focus on provided information, no internet knowledge.
- Evaluation, this module delineates evaluation metrics such as, _faithfulness_, _relevancy_, _correctness_, etc. to evaluate agent's performance and define how accuracte a chatbot is in answering the questions.
  - `:TBD:`
- UI: An UI element to debug and test out Chatbot, more like a playground before exposing it to external system as an API.
  - Used `streamlit` to develop a playground UI.
