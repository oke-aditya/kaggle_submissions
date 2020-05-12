import transformers

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 4
EPOCHS = 10
# ACCUMULATION = 2
BERT_PATH = "../models/bert_base_multilingual_uncased/"
MODEL_PATH = "model.bin"
# TRAINING_FILE = "../data/imdb_data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

