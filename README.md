# Take-Home-Task for Retrieval Assistant
 Machine Learning Internship task at home

 # Approach

 My approach to this problem is that I use embeddings for FAQ questions using a pre-trained model which is downloaded automatically (sentence transformer).

 The way I connect the user query to the embeddings from the json is by doing a cosine similarity check.

 It supports multiple languages also as the model  supports it as well as a confidence score.

# Tools Used
The tools I used are:
- Python3
- Sentence transformers (all-MiniLM-L6-v2) 
- Cosine similarity check


# How to run the project 
First you install our only dependancies:

```bash
pip install sentence-transformers torch```

```bash 
python3 ./main.py```


# Scaling 

This approach can scale in a real product like Verba AI by:
- Using a vector database for efficient similarity search.
- Supporting many more multilingual embeddings.
- Integrating with existing customer support systems.