# Working with HuggingFace
## Understanding the "GitHub" of LLMs: half-day workshop
## Session labs 
## Revision 1.2 - 10/20/24

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Setup and Explore Hugging Face**

**Purpose: In this lab, we'll start to get familiar with the Hugging Face platform and see how to run simple tasks using pre-trained models.**

1. In a browser, go to *https://huggingface.co/models*.

2. Let's search for another simple model to try out. In the search bar, enter the text *DialoGPT*. Look for and select the *microsoft/DialoGPT-medium* model.
  ![model search](./images/dga44.png?raw=true "model search")

3. Let's see how we can quickly get up and running with this model. On the *Model Card* page for the *microsoft/DialoGPT-medium* model, if you scroll down, you'll see a *How to use* section with some code in it. Highlight that code and copy it so we can paste it in a file in our workspace.

![code to use model](./images/dga45.png?raw=true "code to use model")
   
4. Switch back to your codespace and the *diy-gen-ai* directory (if not there). Create a new file named dgpt-med.py (or whatever you want to call it). Paste the code you copied from the model card page into the file. You can create the new file from the terminal using:

```
code dgpt-med.py
```
![adding code](./images/dga46.png?raw=true "adding code")

5. Don't forget to save your file. Now you can run your file by invoking it with python. You'll see it start to download the files associated with the model. This will take a bit of time to run.
```
python dgpt-med.py
```
![running the model](./images/dga47.png?raw=true "running the model")

6. After the model loads, you'll see a *>> User:* prompt. You can enter a prompt or question here, and after some time, the model will provide a response.  **NOTE** This model is small and old and does not provide good responses usually or even ones that make sense. We are using it as a simple, quick demo only.

```
>> User: <prompt here>
```
![running the model](./images/dga48.png?raw=true "running the model")

7. Let's now switch to a different model. Go back to the Hugging Face search and look for *phi3-vision*. Find and select the entry for *microsoft/Phi-3-vision-128k-instruct*.
![finding the phi3-vision model](./images/dga49.png?raw=true "finding the phi3-vision model")

8. Switch to the *Files and versions* page to see the sizes of the files in the Git repository. Note the larger sizes of the model files themselves.
![examining the model files](./images/dga53.png?raw=true "examining the model files")

9. Now, let's see how we can try this model out with no setup on our part. Go back to the *Model card* tab, and scroll down to the *Resources and Technical Documentation* section. Under that, select the entry for *Phi-3 on Azure AI Studio*.
![Invoking model on Azure AI Studio](./images/dga54.png?raw=true "Invoking the model on Azure AI Studio")

10. This will start up a separate browser instance of Azure AI Studio with the model loaded so you can query it. In the prompt area, enter in a prompt to have the AI describe a picture. You can upload one, enter the URL of one on the web, or use the example one suggested below. After you submit your prompt, the model should return a description of the photo. (If you get a response like *"Sorry I can't assist with that."*, refresh the page and try again.)
```
Describe the image at https://media.istockphoto.com/id/1364253107/photo/dog-and-cat-as-best-friends-looking-out-the-window-together.jpg?s=2048x2048&w=is&k=20&c=Do171m5e2DbPIlWDs1JfHn-g8Et_Hxb2AskHg4cRYY4=
```
![Describing an image](./images/dga55.png?raw=true "Describing an image")


<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 2 - Creating a custom pipeline**

**Purpose: In this lab, we’ll see how to interact with Hugging Face pipelines and create a custom pipeline**

1. In our repository, we have a program to do sentiment analysis. The file name is sentiment.py. Open the file either by clicking on [**genai/sentiment.py**](./genai/sentiment.py) or by entering the command below in the codespace's terminal.

```
code sentiment.py
```

2. Notice that it's using a Hugging Face pipeline to do the analysis (see line 5). We've seeded it with some random strings as data to work against. When ready, go ahead and run it with python in the codespace's terminal. In the output, observe which ones it classified as positive and which as negative and the relative scores.

```
python sentiment.py
```

3. Now let's create a custom pipeline. We'll create one that does translation from one language to another and then runs sentiment analysis on the results - basically combining two existing pipelines. Start out by creating a new file for the custom code with the command below.

```
code lab2.py
```

4. Now add the code to import the necessary models and pipelines. Put the following into the new file. In this code, the translator uses a pre-trained model for translating French to English (can be replaced for other languages). And the sentiment_analyzer is a pre-trained sentiment analysis model that works on English text.

```python
from transformers import pipeline

# Load the translation pipeline (for translating text to English)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

# Load the sentiment analysis pipeline (to classify English text)
sentiment_analyzer = pipeline("sentiment-analysis")
```

5. Next, we'll define a custom pipeline function. Add the code below. “* The function takes non-English text (in this case, French), translates it to English, and then runs sentiment analysis on the translated text. * The function returns both the **translated text** and the **sentiment result**.”

```python
def custom_pipeline(text):
    # Step 1: Translate the text to English if it is non-English (assuming French for now)
    translation = translator(text)[0]['translation_text']
    
    # Step 2: Perform sentiment analysis on the translated English text
    sentiment = sentiment_analyzer(translation)
    
    return {"translated_text": translation, "sentiment": sentiment[0]}
```

6. Finally, let's add code to demo our custom pipeline with multiple strings.

```python
# Test the custom pipeline with multiple French inputs
texts = [
    "J'adore ce produit, il est incroyable !",  # Positive sentiment
    "Ce restaurant est horrible, je ne reviendrai jamais."  # Negative sentiment
]

# Process each text through the custom pipeline
for text in texts:
    result = custom_pipeline(text)
    print(f"Original: {text}")
    print(f"Translated: {result['translated_text']}")
    print(f"Sentiment: {result['sentiment']}")
    print()
```

7. Save your changes to lab2.py and then run the code to see it in action.

```
python lab2.py
```
![running the custom pipeline](./images/hug27.png?raw=true "Running the custom pipeline")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 3 - Publishing a custom pipeline on Hugging Face**

**Purpose: In this lab, we’ll publish the custom pipeline we created in the last lab out on Hugging Face**

1. Make sure you are logged in to your Hugging Face account. We need to have an access token to work with. Go to this URL: https://huggingface.co/settings/tokens/new?tokenType=write to create a new token. (Alternatively, you can go to your user Settings, then select Access Tokens, then Create new token, and select Write for the token type.) Select a name for the token and then Create token.

![creating a new token](./images/hug29.png?raw=true "Creating a new token")

2. After you click the Create button, your new token will be displayed on the screen. Make sure to Copy it and save it somewhere you can get to it for the next steps. You will not be able to see it again.

![new token displayed](./images/hug30.png?raw=true "New token displayed")  

3. While we are in the Hugging Face site, go ahead and create a new repository for the custom pipeline. Go to https://huggingface.co/new . (Alternatively, you can click on your profile icon in the top right corner and select New Model from the dropdown.) Then fill out the details of your model. You can just select "mit" for the license and keep the defaults for the remaining items. Then click on the "Create model" button at the bottom.

![creating new repo ](./images/hug32.png?raw=true "Creating new repo") 
![new repo ](./images/hug32.png?raw=true "New repo") 

4. Run the following command to login with your Hugging Face account credentials. This will prompt you for your Hugging Face token, which can be found in your Hugging Face account settings under Access Tokens. 

```
huggingface-cli login
```

5. At the prompt, paste your token. Note that you will not see it, but just paste it and hit Enter. At the prompt about accepting the token as a git credential, you can just respond 'n' since we don't have things setup for that.

 ![token pasted](./images/hug31.png?raw=true "Token pasted")    

6. Clone the repository down from Hugging Face to have it locally.

```
git clone https://huggingface.co/username/model_name
cd model_name
```

7. Create a basic README.md file by running the first command below. Then paste in the remaining contents and save the file.

```
code README.md

---
license: mit
---
# Custom Translation-Sentiment Pipeline

This pipeline translates French text to English and performs sentiment analysis on the translated text.

## Usage:

```python
from custom_pipeline import CustomPipeline

pipeline = CustomPipeline()
result = pipeline("J'adore ce produit, il est incroyable !")
print(result)

```

8. Copy your custom pipeline file from the previous lab to custom-pipe.py into the custom-pipe directory.

```
cp ../custom-pipeline1.py custom-pipe.py
```

9. For this step, you need your Hugging Face token again. Update your git remote url with the username and token with the first command. Then do a git push to get your changes into your Hugging Face repository. If you named your repo "custom-pipe", then that's what you would use for "<repo-name>". As an example, for my case I would set it to
   git remote set-url origin https://techupskills:hf_rest-of-token@huggingface.co/techupskills/custom-pipe

```
git remote set-url origin https://<user_name>:<token>@huggingface.co/<user-name>/<repo-name>
git push
```
10. Now, you can go to the Hugging Face site for your code and try it out. (TO-DO: figure this out)

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 4 - Fine-tuning a model with datasets**

**Purpose: In this lab, we’ll see how to fine tune a model with a dataset using the transformers library**

1. Create a new file named *lab4.py*. (Hint: You can use the command 'code lab4.py'.) In the new file, first import the necessary libraries for loading the model, tokenizer, dataset, and handling training with PyTorch
   
```python

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
``` 

2. Now, let's load a portion of the GLUE SST-2 dataset for sentiment analysis. This dataset contains sentences labeled as positive or negative. We're only going to use a very small percentage of the dataset (1%) to make this runnable in the time we have, but it will be enough to show a difference.
   
```python

train_dataset = load_dataset('glue', 'sst2', split='train[:1%]')
test_dataset = load_dataset('glue', 'sst2', split='validation[:1%]')
```

3. This next section loads a pre-trained model and tokenizer named *DistilBERT*. The tokenizer is used to convert text into tokens, and the model is fine-tuned for sequence classification tasks (binary sentiment analysis in this case).

```python

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

4. Now, we'll create a function to preprocess the dataset. This set of code transforms the text into a numerical format that the model can understand. It adds padding and does truncation where needed to get to a consistent input size.

```python

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
```

5. We still need to convert the tokenized dataset into a format that PyTorch can understand and use. The code below accomplishes this, which is a crucial step for preparing our data to use for training and evaluation.
   
```python

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

6. Let's get a "before picture" by evaluating the model *before* fine-tuning. The eval is done by making predictions on the test dataset and calculating accuracy. This gives us a baseline for comparison.
   
```python

test_dataloader = DataLoader(tokenized_test, batch_size=16)

model.eval()

correct = 0
total = 0

for batch in test_dataloader:
    inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
    labels = batch['label']
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

pre_fine_tune_accuracy = correct / total
print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.2f}')
```

7. The next set of code configures the training process with needed parameters. These include the number of epochs, batch size, logging settings, and output directory. The arguments define how the model is trained.
   
```python

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=10,
    logging_dir='./logs',
    eval_strategy='no',
    save_strategy='no',
    disable_tqdm=True,
)
```

8. Now we initialize the Hugging Face Trainer class. This simplifies the training loop and handles optimization, logging, etc.

```python

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)
```

9. Finally, we can run the fine-tuning process on the pre-trained model using the training dataset. Fine-tuning adapts the model to the specific task (in this case, sentiment classification on the SST-2 dataset).

```python

trainer.train()
```

10. Now, we re-evaluate the model after fine-tuning to measure the performance improvement. This is done again by  comparing predictions to actual labels and calculates the accuracy.

```python

model.eval()

correct = 0
total = 0

for batch in test_dataloader:
    inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
    labels = batch['label']
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

post_fine_tune_accuracy = correct / total
print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.2f}')
```


11. Finally, we want to display the accuracy before and after fine-tuning to quantify the performance improvement.

```python

print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.2f}')
print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.2f}')
```

12. Execute the code to see the fine-tuning happen and the difference before and after. Remember we are only using a very small subset of the dataset, but we are also fine-tuning and testing with the same subset. This will take several minutes to run.

```
python lab4.py
```
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Using datasets with agents and RAG**

**Purpose: In this lab, we’ll see how to use datasets from Hugging Face with other frameworks, agents, and RAG.**

1. In this lab, we'll download a medical dataset, parse it into a vector database, and create an agent with a tool to help us get answers. First,let's take a look at a dataset of information we'll be using for our RAG context. We'll be using a medical Q&A dataset called [**keivalya/MedQuad-MedicalQnADataset**](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset). You can go to the page for it on HuggingFace.co and view some of it's data or explore it a bit if you want. To get there, either click on the link above in this step or go to HuggingFace.co and search for "keivalya/MedQuad-MedicalQnADataset" and follow the links.
   
![dataset on huggingface](./images/rag27.png?raw=true "dataset on huggingface")    

2. Now, let's create the Python file that will pull the dataset, store it in the vector database and invoke an agent with the tool to use it as RAG. First, create a new file for the project.
```
code lab5.py
```

3. Now, add the imports.
```python
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
```

4. Next, we pull and load the dataset.
   
```python
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
data = data.to_pandas()
data = data[0:100]
df_loader = DataFrameLoader(data, page_content_column="Answer")
df_document = df_loader.load()
```

5. Then, we split the text into chunks and load everything into our Chroma vector database.

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1250,
                                      separator="\n",
                                      chunk_overlap=100)
texts = text_splitter.split_documents(df_document)

# set some config variables for ChromaDB
CHROMA_DATA_PATH = "vdb_data/"
embeddings = FastEmbedEmbeddings()  

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(df_document, embeddings, persist_directory=CHROMA_DATA_PATH)
```

6. Set up memory for the chat, and choose the LLM.

```python
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=4, #Number of messages stored in memory
    return_messages=True #Must return the messages in the response.
)

llm = Ollama(model="llama3",temperature=0.0)
```

7. Now, define the mechanism to use for the agent and retrieving data. ("qa" = question and answer) 

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_chroma.as_retriever()
)
```

8. Define the tool itself (calling the "qa" function we just defined above as the tool).
from langchain.agents import Tool

```python
#Defining the list of tool objects to be used by LangChain.
tools = [
   Tool(
       name='Medical KB',
       func=qa.run,
       description=(
           'use this tool when answering medical knowledge queries to get '
           'more information about the topic'
       )
   )
]
```

8. Create the agent using the LangChain project *hwchase17/react-chat*.

```python
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(
   tools=tools,
   llm=llm,
   prompt=prompt,
)

# Create an agent executor by passing in the agent and tools
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               memory=conversational_memory,
                               max_iterations=30,
                               max_execution_time=600,
                               #early_stopping_method='generate',
                               handle_parsing_errors=True
                               )
```

9. Add the input processing loop.

```python
while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    agent_executor.invoke({"input": query})
```

10. Now, **save the file** and run the code.

```
python lab5.py
```

11. You can prompt it with queries related to the info in the dataset, like:
```
I have a patient that may have Botulism. How can I confirm the diagnosis?
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Creating a Sentiment Analysis Web App using Hugging Face and Gradio**

**Purpose: In this lab, we'll create a web-based sentiment analysis application using Hugging Face transformers and Gradio. This app will analyze the sentiment of a given text and classify it as positive, neutral, or negative.**

1. Create a new file (suggested name *app.py*)and open it in the editor. Add the imports for the necessary libraries. These imports bring in the model processing and tokenizer from Hugging Face, tools for numerical calculations, and the Gradio library to build the web interface.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import gradio as gr
```
2. Now, we need to define the pre-trained model path that we want to use for sentiment analysis. We use one which is specifically trained on Twitter data for classifying sentiments.

```python
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

3. Next is loading the tokenizer from the pre-trained model. This converts the text input into a format the model can understand. It ultimately handles breaking sentences into tokens, padding, and converting tokens into numerical data.

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

4. Load the model configuration that contains the parameters for the model. This includes elements such as the number of the classes for classification and the architecture details. This is helpful in getting the right settings in place for the model.

```python
config = AutoConfig.from_pretrained(model_path)
```

5. And load the pre-trained model that will perform the sentiment classification.
```python
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

6. Now we define the function that performs the sentiment analysis. It works by tokenizing the input text and processing it using the model to predict probabilitiies. The probabilities are then converted to the corresponding sentiment labels (Negative, Neutral, Positive).

```python
def sentiment_analysis(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}
    return scores
```

7. We're ready to create the Gradio web interface for the app. This creates the interface, connects the previous function to process the input and displays the sentiment as an output label.

```python
demo = gr.Interface(
    theme=gr.themes.Base(),
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your text here..."),
    outputs="label",
    examples=[
        ["I'm thrilled about the job offer!"],
        ["The weather today is absolutely beautiful."],
        ["I had a fantastic time at the concert last night."],
        ["I'm so frustrated with this software glitch."],
        ["The customer service was terrible at the store."],
        ["I'm really disappointed with the quality of this product."]
    ],
    title='Sentiment Analysis App',
    description='This app classifies a positive, neutral, or negative sentiment.'
)
```

8. Finally, we'll add code to launch the web app.

```python
demo.launch(server_name="0.0.0.0", server_port=9200)
```

9. Now, you're ready to test the app. Run the code below to start it. Then you can input sentences to test the sentiment analysis.

```
python app.py
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 7 - Sharing our app to Hugging Face Spaces**

**Purpose - In this lab, we'll share the Gradio app we created in the last lab to a Hugging Face Space**

1. Make sure you're signed in to your Hugging Face account. Click on your profile icon in the top-right corner and select *New Space*.

![new space](./images/hug34.png?raw=true "New Space")


2. Fill out the form:
    - **Name:** Provide a name for your space (e.g., sentiment-analysis-app).
    - **License:** Choose a license or just choose *MIT*.
    - **Visibility:** Select *Public*.
    - **SDK:** Select *Gradio*.

![complete form](./images/hug23.png?raw=true "Complete form")  

3. You can keep the defaults for the rest of the options. Click on the "Create Space" button to finish creating the new space.

![finish form](./images/hug24.png?raw=true "Finish form")  

4. You'll now be on the screen with guidance for how to upload your Gradio app to the new space. Near the top of the page you'll find the command to clone your repo down. Clone your repository down and then change into the directory.

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analysis-app
cd sentiment-analysis-app
```
  
![clone guidance](./images/hug26.png?raw=true "Clone guidance")  

5. Create a *requirements.txt* file with these contents and save it.

```
transformers
gradio
scipy
numpy
```

6. Copy the *app.py* file from the last lab into the cloned directory. Add it and the *requirements.txt* file to Git and execute the commands from the dialog in your Codespace terminal to get the app put into Spaces.

```bash
cp ../app.py .
git add app.py requirements.txt
git commit -m "Add Gradio sentiment analysis app"
git push
```
7. Once pushed, the Hugging Face platform will automatically build and deploy your Gradio app. You can look at the build log on the site and monitor progress.

8. When the build process is complete and the ap is deployed, you'll see your Gradio app live on the Hugging Face Space. Test it by entering some text in the input box and see the sentiment classification.

9. You can share the URL of your space. It's

```bash
https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analysis-app
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

<p align="center">
**THANKS!**
</p>
