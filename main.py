import json, torch, argparse
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

class FAQAssistant:
    def __init__(self, model_name: str | None = None):
        with open("faq_dataset.json", "r") as file:
            faq_data = json.load(file)

        default_model = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.model = SentenceTransformer(model_name or default_model)
        self.questions = [item['question'] for item in faq_data] 
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
        self.answers = {item['question']: item['answer'] for item in faq_data}


    def retrieve_faq(self, query, top_k=3):
        self.query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(self.query_embedding, self.question_embeddings)[0]
        top_results = torch.topk(similarities, k=top_k)
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            question = self.questions[idx]
            results.append({
                'question': question,
                'answer': self.answers[question], 
                'confidence': float(score)
            })

        return results 
        


    def detect_language(self, text: str) -> str: 
        try:
            code = detect(text)
            return (code or 'en').split('-')[0] 
        except LangDetectException:
            return 'en'


    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        if not text or source_language == target_language: 
            return text
        try:
            return GoogleTranslator(source=source_language, target=target_language).translate(text) 
        except Exception:
            return text
    

    def run_cli(self):
        print("Welcome to the FAQ Assistant!")
        while True:
            user_query = input("Enter your question (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break

            top_faqs = self.retrieve_faq(user_query)

            print("\nTop Relevant FAQs:")
            for idx, faq in enumerate(top_faqs, start=1):
                print(f"{idx}. {faq['question']} (Confidence: {faq['confidence']:.2f})")

            
            user_lang = self.detect_language(user_query)  
            best_answer_en = top_faqs[0]['answer']
            best_answer_out = self.translate_text(best_answer_en, source_language='en', target_language=user_lang)

            print(f"\nBest Answer: {best_answer_out}\n")



def parse_args():
    parser = argparse.ArgumentParser(description="FAQ Assistant")
    parser.add_argument(
        "--model", "-m", 
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    assistant = FAQAssistant(model_name=args.model)
    assistant.run_cli()