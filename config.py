INPUT_PATH = "./sentiment_labelled_sentences/data/"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 128
PLOT_PATH = "./plot/"
EPOCHS = 100
TARGET_COLUMNS = [
    "text",
    "text_remove_puncs_remove_stopwords",
    "text_remove_new_stopwords"
]