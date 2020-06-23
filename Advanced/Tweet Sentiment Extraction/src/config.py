import tokenizers
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 16
EPOCHS = 10
# ACCUMULATION = 2
BERT_PATH = "E:\\Aditya\\AI_DL_ML\Models\\bert_models_pytorch\\bert-base-uncased\\"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "bert-base-uncased-vocab.txt"), lowercase=True
)
