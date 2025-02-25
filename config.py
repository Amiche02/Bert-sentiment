import transformers

class CONFIG:
    # Paths
    BERT_MODEL_NAME = "bert-base-uncased" 
    MODEL_PATH = "model.bin"

    # Device
    DEVICE = "cuda"
    
    # Hyperparameters
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    EPOCHS = 3
    LR = 3e-5
    WARMUP_RATIO = 0.1  # fraction of total steps

    # Tokenizer
    # You can also use AutoTokenizer if you prefer
    TOKENIZER = transformers.BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    
    # Data files
    TRAIN_FILE = "train.csv"
    VALID_FILE = "valid.csv" 
    TEST_FILE = "test.csv"
