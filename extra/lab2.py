from transformers import pipeline

# Load translation and sentiment analysis pipelines
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
sentiment_analyzer = pipeline("sentiment-analysis")

# Define the custom pipeline function
def custom_pipeline(text):
    # Step 1: Translate text to English (from French in this case)
    translation = translator(text)[0]['translation_text']
    
    # Step 2: Perform sentiment analysis on the translated text
    sentiment = sentiment_analyzer(translation)
    
    return {"translated_text": translation, "sentiment": sentiment[0]}

# Test with a single input
result = custom_pipeline("J'adore ce produit, il est incroyable !")
print(f"Single input result: {result}")

# Test with multiple inputs
texts = [
    "J'adore ce produit, il est incroyable !",
    "Ce restaurant est horrible, je ne reviendrai jamais."
]

for text in texts:
    result = custom_pipeline(text)
    print(f"Original: {text}")
    print(f"Translated: {result['translated_text']}")
    print(f"Sentiment: {result['sentiment']}")
    print()
