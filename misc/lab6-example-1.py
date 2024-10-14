from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import gradio as gr

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def sentiment_analysis(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}
    return scores

demo = gr.Interface(
    theme=gr.themes.Base(),
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your text here..."),
    outputs="label",
    examples=[        ["I'm thrilled about the job offer!"],
        ["The weather today is absolutely beautiful."],
        ["I had a fantastic time at the concert last night."],
        ["I'm so frustrated with this software glitch."],
        ["The customer service was terrible at the store."],
        ["I'm really disappointed with the quality of this product."]
    ],
    title='Sentiment Analysis App',
    description='This app classifies a positive, neutral, or negative sentiment.'
)

demo.launch(server_name="0.0.0.0", server_port=9200)
