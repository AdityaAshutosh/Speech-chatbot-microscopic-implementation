#  Q&A chatbot based on Speech Data

A Q&A chatbot using <b>Langchain</b> framework utilizing a SLM like Google's <b>FLAN-T5-Large</b> trained on 783M parameters. 
It is given a speech data from Andrew Huberman's <a href="https://www.youtube.com/watch?v=WDv4AWk0J3U"> podcast clip </a>, 
which is converted into processed text and later passed through a RAG pipeline.

The purpose of utilizing FLAN-T5 was to check the capabilities of SLMs to generate results with RAG pipeline.
We have also used popular embedding model, **all-MiniLM-L6-v2**, for creating vector embeddings and stored it in **FAISS**
library. Speech to text conversion was done using **AssemblyAI** API.

##Project Structure
  - `preprocessing.ipynb`: Speech to text conversion is done using **AssemblyAI** API and further text preprocessing was done using `nltk`.
  -  `embeddings.py` : Embeddings are created by iterating through the JSON response created after preprocessing. Embeddings were created
      using **all-MiniLM-L6-V2** embedding model available in **SentenceTransformer**. Embeddings are added in the FAISS library using FlatL2 
      indexing. Also, includes a function `map_index_to_text` for index-to-text mapping for further retrieval process, by maintaining mapping 
      using `text_segments.pkl`.
  -  `retrieval.py`: Performs query operations from FAISS library as per the user query, fetches relevant embedding indexes, maps them to associated
     text segment using defined function and passes on the input prompt as a combination of retrieved context and user query to **FLAN-T5-Large** SLM
     which generates the final response.
     


## Setup and Installation

To set up and run the project, follow these steps:

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/AdityaAshutosh/Speech-chatbot-microscopic-implementation
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```
   
3. Run `embeddings.py` and `retrieval.py` using `python embeddings.py` and `python retrieval.py`. 
   Type in your query when prompted to, while executing `retrieval.py`.


## Conclusion

This project was created to get an idea of the implementation of methodologies in a RAG pipeline which remain abstracted
while using frameworks like **Langchain** who handle most of the job for you, and that too very efficiently, chaining it
together for multiple usecases. The SLM used here is FLAN-T5-Large, which is trained on merely 700M parameters, compared 
to over 1.7 trillion count for models like GPT-4.
Turns out, small language models are actually not very good with conversational or dialogue based responses, therefore it
is giving inaccurate and overly truncated responses. Embedding model used here is also not very competitive with robust models
like OpenAI embeddings, which can be useful in other use-cases, just not this one.
This application was useful to get an idea of the rough workflow of RAG pipeline.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
