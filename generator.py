import argparse
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME_7B = "OpenAssistant/falcon-7b-sft-top1-696"

tokenizer_7b = AutoTokenizer.from_pretrained(MODEL_NAME_7B)

model_7b = AutoModelForCausalLM.from_pretrained(MODEL_NAME_7B, trust_remote_code=True)




PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"

PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.
{context}"""


END_7B = "\n<|prompter|>{query}<|endoftext|><|assistant|>"

PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>"]
}


def generate_response(model, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        temperature=PARAMETERS["temperature"],
        top_p=PARAMETERS["top_p"],
        repetition_penalty=PARAMETERS["repetition_penalty"],
        top_k=PARAMETERS["top_k"],
        max_new_tokens=PARAMETERS["max_new_tokens"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--step-size', type=int, default=100)

    return parser.parse_args()


def embed(fname, window_size, step_size):
    text = extract_text(fname)
    text = ' '.join(text.split())
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), step_size):
        sentence = text_tokens[i: i+window_size]
        if len(sentence) < window_size:
            break
        sentences.append(sentence)
    paragraphs = [" ".join(s) for s in sentences]
    print(paragraphs)


    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model.max_seq_length = 512
    cross_enc = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    embeddings = model.encode(paragraphs, show_progress_bar=True, convert_to_tensor=True)
    return model, embeddings, cross_enc, paragraphs


def search(query, model, cross_enc, embeddings, paragraphs, top_k):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cuda()
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    cross_input = [[query, paragraphs[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_enc.predict(cross_input)

    for idx in range(len(cross_scores)):
        hits[idx]['cross_score'] = cross_scores[idx]


    results = []
    hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)

    for hit in hits[:5]:
        results.append(paragraphs[hit['corpus_id']].replace('\n', ' '))
    return results







if __name__ == '__main__':
    args = parse_args()
    model, embeddings, cross_enc, paragraphs = embed(args.fname,
                                                     args.window_size,
                                                     args.step_size)

    print(embeddings.shape)

    while True:
        print('\n')

        query = input('Enter query: ')
        results = search(query, model, cross_enc, embeddings, paragraphs, args.top_k)
        print(results)

        query_7b = PREPROMPT + PROMPT.format(context = "\n".join(results))

        query_7b += END_7B.format(query=query)

        response_7b = generate_response(model_7b, tokenizer_7b, query_7b)

        print(response_7b)




