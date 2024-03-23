# from flask import Flask, request, jsonify
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# # from transformers import BertTokenizer,  AutoModelForSequenceClassification
# from flask_cors import CORS 
# import numpy as np
# import torch
# from tqdm import tqdm
# import pandas as pd
# import nltk
# from nltk.util import ngrams
# from nltk.lm.preprocessing import pad_sequence
# from nltk.probability import FreqDist
# from collections import Counter
# from nltk.corpus import stopwords
# import string
# # from plagindataset import run_plagiarism_analysis , vector_database, create_vector_from_text

# nltk.download('punkt')
# nltk.download('stopwords')

# app = Flask(__name__)
# CORS(app)

# # Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# def preprocess_text(text):
#     tokens = nltk.word_tokenize(text.lower())
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
#     return tokens

# def calculate_perplexity(text):
#     encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
#     input_ids = encoded_input[0]

#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits

#     perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
#     return perplexity.item()

# def calculate_burstiness(text):
#     tokens = nltk.word_tokenize(text.lower())
#     word_freq = FreqDist(tokens)
#     repeated_count = sum(count > 1 for count in word_freq.values())
#     burstiness_score = repeated_count / len(word_freq)
#     return burstiness_score

# def get_top_repeated_words(text):
#     tokens = text.split()
#     stop_words = set(stopwords.words('english'))
#     tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

#     word_counts = Counter(tokens)
#     top_words = word_counts.most_common(10)
    
#     return top_words

# # model_path = "bert-base-uncased"
# # tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
# # model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=False, output_hidden_states=True)

# # # Load data and create vector database
# # data_path = "all_sources_metadata_2020-03-13.csv"
# # data_sample = pd.read_csv(data_path, low_memory=False).dropna(subset=['abstract']).reset_index(drop=True)
# # data_sample['paper_id'] = np.random.randint(1, 1000000, size=len(data_sample))
# # vectors = []

# # for text in tqdm(data_sample.abstract.values):
# #     vector = create_vector_from_text(tokenizer, model, text)
# #     vectors.append(vector)

# # data_sample["vectors"] = vectors
# # data_sample["vectors"] = data_sample["vectors"].apply(lambda emb: np.array(emb))
# # data_sample["vectors"] = data_sample["vectors"].apply(lambda emb: emb.reshape(1, -1))

# @app.route('/analyze', methods=['POST'])
# def analyze_text():
#     data = request.get_json()
#     text = data.get('text', '')
    
#     perplexity = calculate_perplexity(text)
#     burstiness_score = calculate_burstiness(text)
#     top_repeated_words = get_top_repeated_words(text)

#     result = {
#         'perplexity': perplexity,
#         'burstiness_score': burstiness_score,
#         'top_repeated_words': top_repeated_words
#     }
    
#     return jsonify(result)



# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tqdm import tqdm
from flask_cors import CORS 
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification , GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence
from nltk.probability import FreqDist
from collections import Counter
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

data_path = "all_sources_metadata_2020-03-13.csv"
model_path = "bert-base-uncased"
sample_size = 100

# Load the data and preprocess it
data = pd.read_csv(data_path, low_memory=False)
data = data.dropna(subset=['abstract']).reset_index(drop=True)
data['paper_id'] = np.random.randint(1, 1000000, size=len(data))
source_data = data.sample(sample_size)[['abstract', 'paper_id']]

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path,output_attentions=False,output_hidden_states=True)

def create_vector_from_text(tokenizer, model, text, MAX_LEN = 510):
    input_ids = tokenizer.encode(
                        text, 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,                           
                   )    

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                              truncating="post", padding="post")
    
    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks    
    attention_mask = [int(i>0) for i in input_ids]
    
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():        
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Extract the embedding.
    vector = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    vector = vector.detach().cpu().numpy()

    return(vector)

def create_vector_database(data):
    # The list of all the vectors
    vectors = []
    
    # Get overall text data
    source_data = data.abstract.values
    
    # Loop over all the comment and get the embeddings
    for text in tqdm(source_data):
        
        # Get the embedding 
        vector = create_vector_from_text(tokenizer, model, text)
        
        #add it to the list
        vectors.append(vector)
    
    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
    data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))
    
    return data


vector_database = create_vector_database(source_data)

def process_document(text):
    """
    Create a vector for given text and adjust it for cosine similarity search
    """
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)

    return text_vect

def is_plagiarism(similarity_score, plagiarism_threshold):
    is_plagiarism = False

    if similarity_score >= plagiarism_threshold :
        is_plagiarism = True

    return float(similarity_score), is_plagiarism


def check_incoming_document(incoming_document):
  # text_lang = detect(incoming_document)
  # language_list = ['de', 'fr', 'el', 'ja', 'ru']

  final_result = incoming_document 

  # if(text_lang == 'en'):
  #   final_result = incoming_document 

  # elif(text_lang not in language_list):
  #   final_result = None

  # else:
  #   # Translate in English
  #   final_result = translate_text(incoming_document, text_lang)

  return final_result

def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3

    # Check the language of the query/incoming text and translate if required.
    document_translation = check_incoming_document(query_text)

    if document_translation is None:
        print("Only the following languages are supported: English, French, Russian, German, Greek and Japanese")
        exit(-1)

    else:
        # Preprocess the document to get the required vector for similarity analysis
        query_vect = process_document(document_translation)

        # Run similarity Search
        data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
        data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

        similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N + 1]
        formated_result = similar_articles[["abstract", "paper_id", "similarity"]].reset_index(drop=True)

        similarity_score = float(formated_result.iloc[0]["similarity"])
        most_similar_article = formated_result.iloc[0]["abstract"]
        is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

        plagiarism_decision = {'similarity_score': similarity_score,
                               'is_plagiarism': is_plagiarism_bool,
                               'most_similar_article': most_similar_article,
                               'article_submitted': query_text
                               }

        return plagiarism_decision

    
@app.route('/detect_plagiarism', methods=['POST'])
def detect_plagiarism():
    request_data = request.get_json()
    query_text = request_data.get('text', '')
    plagiarism_threshold = request_data.get('plagiarism_threshold', 0.8)

    analysis_result = run_plagiarism_analysis(query_text, vector_database, plagiarism_threshold)
    
    return jsonify(analysis_result)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    perplexity = calculate_perplexity(text)
    burstiness_score = calculate_burstiness(text)
    top_repeated_words = get_top_repeated_words(text)

    result = {
        'perplexity': perplexity,
        'burstiness_score': burstiness_score,
        'top_repeated_words': top_repeated_words
    }
    
    return jsonify(result)

def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def get_top_repeated_words(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    
    return top_words

if __name__ == '__main__':
    app.run(debug=True)
