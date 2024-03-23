import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(data_path, sample_size):

    # Read the data from specific path
    data = pd.read_csv(data_path, low_memory=False)

    # Drop articles without Abstract
    data = data.dropna(subset=['abstract']).reset_index(drop=True)

    # Generate a random paper_id for each row
    data['paper_id'] = np.random.randint(1, 1000000, size=len(data))

    # Get "sample_size" random articles
    data_sample = data.sample(sample_size)[['abstract', 'paper_id']]

    return data_sample

data_path = "all_sources_metadata_2020-03-13.csv"
source_data = preprocess_data(data_path, 100)

print("Sample data:\n", source_data)

model_path = "bert-base-uncased"

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

  if(similarity_score >= plagiarism_threshold):
    is_plagiarism = True

  return is_plagiarism


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

    top_N=3

    # Check the language of the query/incoming text and translate if required. 
    document_translation = check_incoming_document(query_text)

    if(document_translation is None):
      print("Only the following languages are supported: English, French, Russian, German, Greek and Japanese")
      exit(-1)

    else:
      # Preprocess the document to get the required vector for similarity analysis
      query_vect = process_document(document_translation)
      
      # Run similarity Search
      data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
      data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

      similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
      formated_result = similar_articles[["abstract", "paper_id", "similarity"]].reset_index(drop = True)

      similarity_score = formated_result.iloc[0]["similarity"] 
      most_similar_article = formated_result.iloc[0]["abstract"] 
      is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

      plagiarism_decision = {'similarity_score': similarity_score, 
                             'is_plagiarism': is_plagiarism_bool,
                             'most_similar_article': most_similar_article, 
                             'article_submitted': query_text
                            }

      return plagiarism_decision
     
# new_incoming_text = source_data.iloc[0]['abstract']
new_incoming_text = "The COVID-19 pandemic, caused by the novel coronavirus SARS-CoV-2, has had profound global impacts since its emergence in late 2019. This highly contagious virus spreads primarily through respiratory droplets, leading to a wide range of symptoms from mild respiratory issues to severe pneumonia and organ failure. The pandemic has strained healthcare systems, economies, and societies worldwide, prompting unprecedented public health responses such as lockdowns and travel restrictions. Rapid vaccine development efforts have offered hope for controlling the spread, but challenges remain, including misinformation and vaccine distribution inequities. Addressing these challenges requires continued research, collaboration, and public health measures to mitigate the impact of COVID-19."

# Run the plagiarism detection
analysis_result = run_plagiarism_analysis(new_incoming_text, vector_database, plagiarism_threshold=0.8)

analysis_result
