import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import gradio as gr

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def download_and_process_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    corpus = " ".join(lines)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    return corpus


def get_corpus(url):
    file_path = "corpus.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return download_and_process_url(url)


def build_index(corpus):
    sentences = corpus.split('. ')
    embeddings = model.encode(sentences)
    embeddings_array = np.array(embeddings)
    d = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)

    return sentences, index


def get_relevant_sentences(query, sentences, index):
    query_embedding = model.encode([query])
    k = 5
    distances, indices = index.search(query_embedding, k)

    return [sentences[idx] for idx in indices[0]], distances[0]


def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer


def answer_question(url, question):
    corpus = get_corpus(url)
    sentences, index = build_index(corpus)

    relevant_sentences, distances = get_relevant_sentences(question, sentences, index)

    answers = []
    for sentence in relevant_sentences:
        answer = get_answer(question, sentence)
        answers.append(f"Context: {sentence}\nAnswer: {answer}")

    return "\n\n".join(answers)


iface = gr.Interface(
    fn=answer_question,
    inputs=[gr.Textbox(label="Enter URL"), gr.Textbox(label="Enter your question")],
    outputs=gr.Textbox(label="Answer"),
    live=True
)

iface.launch()

# don't use it for better inference time (but u can)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("") # EleutherAI/gpt-j-6B              for example
model = AutoModelForCausalLM.from_pretrained("") # EleutherAI/gpt-j-6B              for example

def generate_detailed_answer(question, context, qa_answers):
    prompt = f'
    Question: {question}
    Context: {context}
    QA Answers:
    {qa_answers}

    Generate a detailed and comprehensive answer using information from the context and QA answers.
    '

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=300, temperature=0.7, num_return_sequences=1)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

for i, idx in enumerate(indices[0]):
    context = sentences[idx]
    answer = get_answer(query, context)
    detailed_answer = generate_detailed_answer(query, context, answer)
    print(f"Detailed answer {i + 1}: {detailed_answer}")
"""