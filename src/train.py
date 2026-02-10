import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
from sklearn.model_selection import train_test_split

def generate_synthetic_data(ods_orgs, num_samples=1000):
    """Generate synthetic training data by inserting org names into sentences."""
    sentences = []
    labels = []
    # Simple templates
    templates = [
        "The meeting was attended by representatives from {}.",
        "{} presented their quarterly report.",
        "Discussions included input from {}.",
        "Collaboration with {} is essential.",
        "{} has approved the new policy."
    ]
    for _ in range(num_samples):
        org = list(ods_orgs)[torch.randint(0, len(ods_orgs), (1,))[0].item()]
        template = templates[torch.randint(0, len(templates), (1,))[0].item()]
        sentence = template.format(org)
        # Label tokens: B-ORG for start, I-ORG for inside, O for outside
        tokens = sentence.split()
        label = ['O'] * len(tokens)
        org_tokens = org.split()
        for i, token in enumerate(tokens):
            if token.lower() in [ot.lower() for ot in org_tokens]:
                if i == 0 or label[i-1] == 'O':
                    label[i] = 'B-ORG'
                else:
                    label[i] = 'I-ORG'
        sentences.append(tokens)
        labels.append(label)
    return sentences, labels

def tokenize_and_align_labels(sentences, labels, tokenizer, label_to_id):
    """Tokenize and align labels for BERT."""
    tokenized_inputs = tokenizer(sentences, truncation=True, is_split_into_words=True)
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label[word_idx] == label_to_id['I-ORG'] else -100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def train_ner_model(ods_path, model_save_path="models/ner_model"):
    """Fine-tune NER model."""
    # Load ODS
    df = pd.read_csv(ods_path)
    ods_orgs = set(df['Organisation Name'].tolist())

    # Generate data
    sentences, labels = generate_synthetic_data(ods_orgs)

    # Label to id
    label_list = ['O', 'B-ORG', 'I-ORG']
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    # Convert labels to ids
    labels = [[label_to_id[l] for l in lab] for lab in labels]

    # Split
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.1)

    # Tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    train_encodings = tokenize_and_align_labels(train_sentences, train_labels, tokenizer, label_to_id)
    val_encodings = tokenize_and_align_labels(val_sentences, val_labels, tokenizer, label_to_id)

    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save label mappings
    with open(f"{model_save_path}/label_config.json", "w") as f:
        json.dump({"id2label": id_to_label, "label2id": label_to_id}, f)

if __name__ == "__main__":
    train_ner_model("data/ods_organisations.csv")
