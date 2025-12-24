import json, torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

with open("faq_dataset.json", "r") as file:
    faq_data = json.load(file)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

questions = [item['question'] for item in faq_data] 
answers = {item['question']: item['answer'] for item in faq_data}


question_embeddings = model.encode(questions, convert_to_tensor=True)

def retrieve_faq(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        question = questions[idx]
        results.append({
        'question': question,
            'answer': answers[question],
            'confidence': float(score)
        })

    return results 


def detect_language(text: str) -> str: 
    try:
        code = detect(text)
        return (code or 'en').split('-')[0] 
    except LangDetectException:
        return 'en'


def translate_text(text: str, source_language: str, target_language: str) -> str:
    if not text or source_language == target_language: 
        return text
    try:
        return GoogleTranslator(source=source_language, target=target_language).translate(text) 
    except Exception:
        return text



if __name__ == "__main__":
    print("Welcome to the FAQ Assistant!")
    while True:
        user_query = input("Enter your question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        top_faqs = retrieve_faq(user_query)

        print("\nTop Relevant FAQs:")
        for idx, faq in enumerate(top_faqs, start=1):
            print(f"{idx}. {faq['question']} (Confidence: {faq['confidence']:.2f})")

        
        user_lang = detect_language(user_query) 
        best_answer_en = top_faqs[0]['answer']
        best_answer_out = translate_text(best_answer_en, source_language='en', target_language=user_lang)

        print(f"\nBest Answer: {best_answer_out}\n")