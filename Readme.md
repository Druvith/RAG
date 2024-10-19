# **RAG using open-source models**
- Creating a RAG based system (retrieval + generation) using open-source models

------

## What is RAG?
* RAG (Retrieval-Augmented Generation) is an framework that combines the strengths of information retrieval systems (such as search and vector databases) with the generative and reasoning capabilities of Large language models. By combining your data with LLMs grounded generation is more accurate and relevant to your specific needs.
------

Current stack:
   - [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) to encode the text
   - A vanilla implementation mimicking the functionality of a vector database. (indexing vector, retrieving similar vectors + scores, deleting the index)
   - Llama 3.2 1B for generation (?)

------
