# Web Question Answering with FAISS and BERT
This project enables users to extract information and answer questions from web pages using a combination of several NLP models and techniques.

# Why all-MiniLM-L6-v2?
- Speed and accuracy
![Why](https://github.com/Yoniqueeml/Ragnetic_QA/blob/master/WhyMiniLm6.png)

# Gradio interface
![Gradio interface](https://github.com/Yoniqueeml/Ragnetic_QA/blob/master/GradioResults.png)

# Features
- Web Scraping
- Sentence Embeddings
- FAISS Indexing
- Gradio Interface

# Dependencies
To run this project, you need to install the following Python libraries:

- ```pip install requests beautifulsoup4 sentence-transformers faiss-cpu torch transformers gradio``` 

#
How it Works
- Download and Process Web Content
Enter a URL in the Gradio interface, which the system scrapes using requests and BeautifulSoup.
The content of the page is processed, extracting the text and saving it into a file named corpus.txt.

- Build the FAISS Index
The text from the URL is split into individual sentences.
Each sentence is then encoded into sentence embeddings using the SentenceTransformer model.
These embeddings are added to a FAISS index to facilitate fast similarity searches.

- Search for Relevant Sentences
When a user asks a question, the system encodes the query into a sentence embedding.
The FAISS index is searched to find the most relevant sentences based on cosine similarity.

- Answer the Question
The top relevant sentences are passed to a fine-tuned BERT model (bert-large-uncased-whole-word-masking-finetuned-squad) for question answering.
The BERT model outputs the answer based on the context provided by the retrieved sentences.
Display Results:

The Gradio interface displays the answer along with the context it was derived from, making it clear how the answer was formed.
# How to Use
- Launch the Interface
After installing the dependencies, run the Python script to launch the Gradio interface

- Ask a Question:
Input a URL from which you'd like to extract information and ask a question related to that webpage.
The system will return an answer based on the context extracted from the page.

# Example
URL: https://en.wikipedia.org/wiki/Artificial_intelligence    
 Question: What is Artificial Intelligence?
