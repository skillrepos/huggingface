from transformers import Pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

class TranslateAndSentimentPipeline(Pipeline):
    def __init__(self, **kwargs):
         # Load translation model and tokenizer
        self.translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        # Load sentiment analysis model and tokenizer
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        super().__init__(model=self.translator_model, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, **kwargs):
        # Tokenize the English text for translation
        translation_inputs = self.translator_tokenizer.encode(inputs, return_tensors="pt")
        return {"translation_inputs": translation_inputs, "original_text": inputs}

    def _forward(self, model_inputs, **kwargs):
        # Translate English text to French
        translation_outputs = self.translator_model.generate(model_inputs["translation_inputs"])
        # Decode the translated French text
        french_text = self.translator_tokenizer.decode(translation_outputs[0], skip_special_tokens=True)
        # Tokenize the French text for sentiment analysis
        sentiment_inputs = self.sentiment_tokenizer(french_text, return_tensors="pt")
        # Perform sentiment analysis
        sentiment_outputs = self.sentiment_model(**sentiment_inputs)
        return {
            "original_text": model_inputs["original_text"],
            "french_text": french_text,
            "sentiment_outputs": sentiment_outputs
        }

    def postprocess(self, model_outputs, **kwargs):
        # Get sentiment scores
        probs = F.softmax(model_outputs["sentiment_outputs"].logits, dim=1)
        # Get the predicted sentiment label
        sentiment_score = torch.argmax(probs, dim=1).item()
        # Map the sentiment score to labels (1 to 5 stars)
        sentiment_label = sentiment_score + 1  # Model predicts labels from 0 to 4
        return {
            "original_text": model_outputs["original_text"],
            "translated_text": model_outputs["french_text"],
            "sentiment": {
                "label": f"{sentiment_label} stars",
                "score": probs[0][sentiment_score].item()
            }
        }
