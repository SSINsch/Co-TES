from transformers import BertForSequenceClassification


class BertClassifier(BertForSequenceClassification):
    alg_name = 'BERT'
