import spacy
import datasets
import vec2text.data_helpers as data_helpers
from datasets import DatasetDict, Dataset, load_from_disk


# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to process a single dataset
def process_dataset(dataset, dataset_name):
    new_data = []
    for record in dataset:  # Assuming each record has a "text" field
        sentence = record["text"]
        doc = nlp(sentence)
        # Extract person entities
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if person_names:
            # sentence, labels, dataset
            new_data.append([sentence,", ".join(person_names),dataset_name])
    return new_data

# Function to process a DatasetDict
def process_dataset_dict(dataset_dict):
    headers = ["sentences", "labels", "dataset"]
    processed_data = []
    for key, dataset in dataset_dict.items():
        print(key, flush=True)
        curr_data = process_dataset(dataset, key)
        if curr_data:
            print(curr_data[0], flush=True)
        processed_data.extend(curr_data)    
    # Convert to DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict({header: [row[i] for row in processed_data] for i, header in enumerate(headers)})
    })
    # Save the dataset to disk
    dataset_path = "/share/u/koyena/datasets/person_finder_dataset"
    dataset.save_to_disk(dataset_path)
    print(f"DatasetDict saved to {dataset_path}", flush=True)
    # with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(processed_data)

# Example DatasetDict
data = data_helpers.load_standard_val_datasets()

# Load the DatasetDict from disk
loaded_dataset = load_from_disk("/share/u/koyena/datasets/person_finder_dataset")
print("Loaded DatasetDict:")

# Access and display the 'train' split
train_split = loaded_dataset["train"]
print(train_split)
print(train_split['labels'])
print(train_split.to_pandas())

# Process the DatasetDict
# processed_dataset_dict = process_dataset_dict(data)
# process_dataset_dict(data)

# Output the processed DatasetDict
#print(processed_dataset_dict)

