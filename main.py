import json, torch
from sentence_transformers import SentenceTransformer, util

with open("faq_dataset.json", "r") as file:
    faq_data = json.load(file)

model = SentenceTransformer('all-MiniLM-L6-v2')

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
        
        print(f"\nBest Answer: {top_faqs[0]['answer']}\n")