from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
nltk.download('punkt')
nltk.download('taggers/averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


#file_path = "results/results_gradients/val-text-preds.csv"
#file_path = "results/results_gradients/combined_val_text_preds.csv"
root_file_path = "results/results_gradients/"
file_path_dict = {
    "ag_news": root_file_path + "ag_news_eval_output.csv",
    "anthropic_toxic_prompts": root_file_path + "anthropic_toxic_prompts_eval_output.csv",
    "arxiv": root_file_path + "arxiv_eval_output.csv",
#     "one_million_instructions_train": root_file_path + "one_million_instructions_train_1000_eval_output.csv",
      "one_million_instructions_train_10k": root_file_path + "one_million_instructions_train_10k_eval_output.csv",
    "one_million_instructions_val": root_file_path + "one_million_instructions_val_eval_output.csv",
    "wikibio": root_file_path + "wikibio_eval_output.csv",
    "xsum_doc": root_file_path + "xsum_doc_eval_output.csv",
    "xsum_summ": root_file_path + "xsum_summ_eval_output.csv",
    "python_code_alpaca": root_file_path + "python_code_alpaca_eval_output.csv"
}
df_dict = {}
for key, value in file_path_dict.items():
    print(f"reading {key}'s file", flush=True)
    df_dict[key] = pd.read_csv(value)

# Load a pre-trained transformer model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute sentence embeddings
def compute_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

# Function to calculate similarity score
def similarity_score(sentence1, sentence2):
    embeddings = compute_embeddings([sentence1, sentence2])
    sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return sim[0][0]

for key, df in df_dict.items():
    
    # Threshold for semantic matching
    similarity_threshold = 0.8
    
    # Initialize counters
    TP = 0
    TP_cos = 0
    FP = 0
    FN = 0
    FP_cos = 0
    FN_cos = 0
    # Evaluate each row
    for _, row in df.iterrows():
        original = row["Original"]
        decoded = row["Decoded"]
        
        if original == decoded:  # Exact match criterion
            TP += 1
            TP_cos += 1
        else:
            FP += 1
            FN += 1
            # Compute semantic similarity
            sim_score = similarity_score(original, decoded)
            if sim_score >= similarity_threshold:
                TP_cos += 1
            else:
                FP_cos += 1
                FN_cos += 1
    
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate precision and recall
    precision_cos = TP_cos / (TP_cos + FP_cos) if (TP_cos + FP_cos) > 0 else 0
    recall_cos = TP_cos / (TP_cos + FN_cos) if (TP_cos + FN_cos) > 0 else 0
    print(f"Key: {key}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision with semantic match: {precision_cos:.2f}")
    print(f"Recall with semantic match: {recall_cos:.2f}")
    print("---------------------------------------", flush=True)


# Function to calculate matching token length
def calculate_matching_length(row):
    original_tokens = word_tokenize(row['Original'])
    decoded_tokens = word_tokenize(row['Decoded'])
    correct_tokens = [tok for tok, dec_tok in zip(original_tokens, decoded_tokens) if tok == dec_tok]
    return len(correct_tokens), len(original_tokens)

# Function to extract person names using NER
def extract_person_names(sentence):
    if not isinstance(sentence, str) or sentence.strip() == "":
        return set()
    chunked = ne_chunk(pos_tag(word_tokenize(sentence)))
    person_names = set()
    for chunk in chunked:
        if isinstance(chunk, Tree) and chunk.label() == 'PERSON':
            name = " ".join(c[0] for c in chunk)
            person_names.add(name)
    return person_names


for key, data in df_dict.items():
# Extract person names from both columns
    output_file_path = key + "_person_instances.csv"
    data['Original_Names'] = data['Original'].apply(extract_person_names)
    data['Decoded_Names'] = data['Decoded'].apply(extract_person_names)
    data[['Original', 'Decoded', 'Original_Names', 'Decoded_Names']].to_csv(output_file_path, index=False)
    
    # Calculate precision and recall
    true_positive = 0
    total_original_names = 0
    total_decoded_names = 0
    
    for _, row in data.iterrows():
        original_names = row['Original_Names']
        decoded_names = row['Decoded_Names']
        
        true_positive += len(original_names & decoded_names)
        total_original_names += len(original_names)
        total_decoded_names += len(decoded_names)
    
    precision = true_positive / total_decoded_names if total_decoded_names > 0 else 0
    recall = true_positive / total_original_names if total_original_names > 0 else 0
    
    # Output the results
    print(f"Key: {key}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}", flush=True)


# file_path = "results/results-gradients/val-text-preds.csv"
# df = pd.read_csv(file_path)

# # Load a pre-trained model from SentenceTransformers
# model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as needed
# tokenizer = model.tokenizer

# # Compute embeddings
# embeddings1 = model.encode(df["Original"].tolist(), convert_to_tensor=True)
# embeddings2 = model.encode(df["Decoded"].tolist(), convert_to_tensor=True)

# # Compute token lengths for each sentence
# df["tokens_original"] = df["Original"].apply(lambda x: len(tokenizer.tokenize(x)))
# df["tokens_decoded"] = df["Decoded"].apply(lambda x: len(tokenizer.tokenize(x)))

# # Compute cosine similarity
# similarities = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
# diagonal_similarities = np.diagonal(similarities)  # Get pairwise similarities

# # Add similarity scores to the DataFrame
# df["similarity"] = diagonal_similarities

# # Sort the DataFrame by similarity in descending order
# df_sorted = df.sort_values(by="similarity", ascending=False)

# # Display results
# print("Sentence pairs sorted by similarity (most to least):")
# for _, row in df_sorted.iterrows():
#     print(f"Pair: ({row['Original']}, {row['Decoded']})\nSimilarity: {row['similarity']:.4f}\n")


# model_name = "nvidia/NV-Embed-v2"
# model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
# max_length = 32768
# model.tokenizer.padding_side="right"

# # Optimized batch processing
# def batch_process(sentences, model, batch_size=50):
#     embeddings = []
#     for i in range(0, len(sentences), batch_size):
#         batch = sentences[i:i+batch_size]
#         batch_embeddings = model.encode(batch, instruction="", max_length=max_length)
#         batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
#         embeddings.append(batch_embeddings.cpu().numpy())
#         print(f"Done with batch {i}", flush=True)
#     return np.vstack(embeddings)


# def get_embeddings(sentences, model):
#     embeddings = model.encode(sentences, instruction="", max_length=max_length)
#     embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
#     return embeddings

# # Compute embeddings for original and decoded sentences
# original_embeddings = batch_process(df["Original"].tolist(), model)
# decoded_embeddings = batch_process(df["Decoded"].tolist(), model)

# # Compute cosine similarity for each pair
# df["similarity"] = [
#     cosine_similarity([original], [decoded])[0, 0]
#     for original, decoded in zip(original_embeddings, decoded_embeddings)
# ]

# # Sort DataFrame by similarity (optional)
# df_sorted = df.sort_values(by="similarity", ascending=False)

# # Display results
# print(df_sorted)


# file_path = "results/results-gradients/val-text-preds.csv"
# df = pd.read_csv(file_path)

# # Load pre-trained GPT-J model and tokenizer
# model_name = "nvidia/NV-Embed-v2"  # You can replace with other transformer models
# model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
# max_length = 32768
# model.tokenizer.padding_side="right"
# # Function to compute sentence embeddings



# # Compute embeddings for original and decoded sentences
# original_embeddings = get_embeddings(df["Original"].tolist(), model)
# decoded_embeddings = get_embeddings(df["Decoded"].tolist(), model)

# # Compute cosine similarity for each pair
# df["similarity"] = [
#     cosine_similarity([original], [decoded])[0, 0]
#     for original, decoded in zip(original_embeddings, decoded_embeddings)
# ]

# # Sort DataFrame by similarity (optional)
# df_sorted = df.sort_values(by="similarity", ascending=False)

# # Display results
# print(df_sorted)
