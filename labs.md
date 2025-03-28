# Working with HuggingFace
## Understanding the "GitHub" of LLMs: half-day workshop
## Session labs 
## Revision 2.6 - 03/27/25

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Exploring Hugging Face**

**Purpose: In this lab, we'll start to get familiar with the Hugging Face platform and see how to run simple tasks using pre-trained models.**
</br></br></br>
1. In a browser, go to *https://huggingface.co/models*.
</br></br></br>
2. Let's search for a simple model to try out. In the search bar, enter the text *DialoGPT*. Look for and select the *microsoft/DialoGPT-medium* model. (Make sure not to select the *small* one.)
  ![model search](./images/hug48.png?raw=true "model search")
</br></br></br>
3. Let's see how we can quickly get up and running with this model. On the *Model Card* page for the *microsoft/DialoGPT-medium* model, if you scroll down, you'll see a *How to use* section with some code in it. Use the *Copy button* in the upper right or highlight the code and copy it so we can paste it in a file in our workspace.

![code to use model](./images/hug49.png?raw=true "code to use model")
</br></br></br>  
4. Switch back to your codespace. Create a new file named dgpt-med.py (or whatever you want to call it). Paste the code you copied from the model card page into the file. You can create the new file from the terminal using:

```bash
code dgpt-med.py
```
![adding code](./images/hug50.png?raw=true "adding code")
</br></br></br>
5. Save your file. Now you can run your file by invoking it with python. You'll see it start to download the files associated with the model. This will take a bit of time to run.
```bash
python dgpt-med.py
```
![running the model](./images/hug51.png?raw=true "running the model")
</br></br></br>
6. After the model loads, you'll see a *>> User:* prompt. (If the model doesn't seem to be finishing loading, try clicking in the terminal and hitting Enter.) You can enter a prompt or question here, and after some time, the model will provide a response.  **NOTE** This model is small and old and does not provide good responses usually or even ones that make sense. We are using it as a simple, quick demo only. (When done, you can exit out with CTRL-D.)

```python
>> User: <prompt here>
```
![running the model](./images/hug52.png?raw=true "running the model")
</br></br></br>
7. Let's now switch to a different model. Go back to the Hugging Face search and look for *phi-4-*. Find and select the entry for *microsoft/Phi-4-multimodal-instruct*.
![finding the phi4-vision model](./images/hug66.png?raw=true "finding the phi4-vision model")
</br></br></br>

8. Now, let's see how we can try this model out with no setup on our part. On the *Model card* tab, under the *Model Summary* section, find the *Try it on* line and click on the *GitHub* link. 
![Trying model on GitHub](./images/hug67.png?raw=true "Trying model on GitHub")
</br></br></br>
9. This will open up a GitHub link with the model loaded so you can query it. In the prompt area, enter in a prompt to have the AI describe a picture. You can upload one, enter the URL of one on the web, or use the example one suggested below. Submit your prompt by hitting *Enter* or clicking on the *Send Now* (arrow) button.  After this, the model should return a description of the photo. 
```
Describe the image at https://media.istockphoto.com/id/1364253107/photo/dog-and-cat-as-best-friends-looking-out-the-window-together.jpg?s=2048x2048&w=is&k=20&c=Do171m5e2DbPIlWDs1JfHn-g8Et_Hxb2AskHg4cRYY4=
```
![Describing an image](./images/hug68.png?raw=true "Describing an image")
</br></br></br>
10. After you are done, you can switch back to *https://huggingface.co* for the next lab.
<p align="center">
**[END OF LAB]**
</p>
</br></br>


**Lab 2 - Demonstrating how to use Hugging Face in programming**

**Purpose: In this lab, we’ll construct a Python program that demonstrates how to use entities from HuggingFace.co to accomplish a meaningful task.**
</br></br></br>
1. In this lab, we'll create a program that performs sentiment analysis on movie reviews from the IMDb dataset and summarizes negative reviews. It leverages multiple features from Hugging Face, including the datasets library and pre-trained models from the transformers library. To start, back in your codespace or local environment, create a new file for our code.

```bash
code lab2.py
```
</br></br></br>
2. First, import the necessary libraries from Hugging Face that we need. These are:

- datasets.load_dataset: Loads datasets hosted on HuggingFace.co.
- transformers.pipeline: Provides easy-to-use interfaces for various NLP tasks using pre-trained models.

Add the code below into the *lab2.py* file.

```python
# Import necessary libraries from Hugging Face
from datasets import load_dataset           # To load datasets from HuggingFace.co
from transformers import pipeline           # For easy inference using pre-trained models
```
</br></br></br>
3. Now, we load the IMDb dataset. (This is a widely used dataset for sentiment analysis tasks.) For purposes of the lab, we'll just select a sample of 10 reviews to look at. Add the code below.

```python
# Load the IMDb dataset from Hugging Face Datasets
# This dataset contains 50,000 movie reviews labeled as positive or negative
imdb_dataset = load_dataset('imdb')

# Select a small sample of reviews for demonstration purposes
sample_reviews = imdb_dataset['test'].select(range(10))  # Selecting first 10 reviews
```
</br></br></br>
4. Now, we'll initialize two pipelines - one for sentiment analysis and one for summarization. The initialization calls will automatically download and load a pre-trained model that can be used for sentiment analysis or summarization, respectively.

```python
# Initialize a sentiment-analysis pipeline using a pre-trained model
sentiment_analyzer = pipeline('sentiment-analysis')

# Initialize a summarization pipeline using a pre-trained model
summarizer = pipeline('summarization')
```
</br></br></br>
5. Next, we'll add code to process the reviews. Here's what the sentiment analysis part does:

 - Sentiment Analysis:
   - We analyze the sentiment of each review.
   - We truncate the text to the first 512 tokens to adhere to model input limitations and improve performance.
   - The result is a dictionary containing the label ('POSITIVE' or 'NEGATIVE') and a confidence score.

Add the code below.

```python
# Iterate over the sample reviews and perform sentiment analysis and summarization
for idx, review in enumerate(sample_reviews):
    text = review['text']  # The review text
    # Perform sentiment analysis
    sentiment = sentiment_analyzer(text[:512])[0]  # Truncate text to 512 tokens for performance
    
    # Print the review index and predicted sentiment
    print(f"Review #{idx + 1} Sentiment: {sentiment['label']} (Score: {sentiment['score']:.4f})")
    
```

6. And, now add the code to do the summarization if the review is negative.
   
 - Conditional Summarization:
   - If a review is negative, we generate a summary of the review.
   - We truncate the text to the first 1024 tokens to prevent errors due to model input size limitations.
   - We specify max_length and min_length to control the length of the summary.
  
Add the code below with the printing of the results.

```python
# If the sentiment is negative, provide a summary of the review
    if sentiment['label'] == 'NEGATIVE':
        # Summarize the review text (truncate to 1024 tokens to avoid errors)
        summary = summarizer(text[:1024], max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        print(f"Summary of Negative Review #{idx + 1}:")
        print(summary)
    
    print("-" * 80)  # Separator between reviews
```
</br></br></br>
7. Save your changes and run the file. After running it, you should see output for the first 10 reviews and summarizations for ones that are determined to be negative. Note that the first time you run it, it will download the required pre-trained models automatically from HuggingFace.co.

```
python lab2.py
```

![running the example](./images/hug57.png?raw=true "Running the example")

</br></br></br>
8. This example demonstrates several features from Hugging Face:
- Loading Datasets with datasets Library:
  - Showcasing how to load and manipulate datasets hosted on HuggingFace.co using the datasets library.
    
- Using Pre-trained Models with transformers Pipeline:
  - Demonstrating using pre-trained models for inference tasks like sentiment analysis and summarization, without deep diving into the model architectures.
    
- Combining Multiple NLP Tasks:
  - Performing both sentiment analysis and summarization, showing how to chain different NLP tasks together.
   
- Handling Model Input Limitations:
  - Illustrating how to manage input sizes (e.g., truncating text) to comply with model constraints.
   
- Customizing Pipeline Parameters:
  - Showing how to adjust parameters (like max_length, min_length, and do_sample) to control model output.

- Interpreting Model Outputs:
  - Parsing and using the outputs of the pipelines, such as extracting labels and confidence scores.

- Efficient Data Processing:
  - Using dataset selection and iteration to process and analyze data samples efficiently.
<p align="center">
**[END OF LAB]**
</p>
</br></br>


**Lab 3 - Fine-tuning a model with datasets**

**Purpose: In this lab, we’ll see how to fine tune a model with a dataset using the transformers library**
</br></br></br>
1. Create a new file named *lab3.py*. (Hint: You can use the command 'code lab3.py'.) In the new file, first import the necessary libraries for loading the model, tokenizer, dataset, and handling training with PyTorch
   
```python

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
``` 
</br></br></br>
2. Now, let's load a portion of the GLUE SST-2 dataset for sentiment analysis. This dataset contains sentences labeled as positive or negative. We're only going to use a very small percentage of the dataset (1%) to make this runnable in the time we have, but it will be enough to show a difference.
   
```python

train_dataset = load_dataset('glue', 'sst2', split='train[:1%]')
test_dataset = load_dataset('glue', 'sst2', split='validation[:1%]')
```
</br></br></br>
3. This next section loads a pre-trained model and tokenizer named *DistilBERT*. The tokenizer is used to convert text into tokens, and the model is fine-tuned for sequence classification tasks (binary sentiment analysis in this case).

```python

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
</br></br></br>
4. Now, we'll create a function to preprocess the dataset. This set of code transforms the text into a numerical format that the model can understand. It adds padding and does truncation where needed to get to a consistent input size.

```python

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
```
</br></br></br>
5. We still need to convert the tokenized dataset into a format that PyTorch can understand and use. The code below accomplishes this, which is a crucial step for preparing our data to use for training and evaluation.
   
```python

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```
</br></br></br>
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
</br></br></br>
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
</br></br></br>
8. Now we initialize the Hugging Face Trainer class. This simplifies the training loop and handles optimization, logging, etc.

```python

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)
```
</br></br></br>
9. Finally, we can run the fine-tuning process on the pre-trained model using the training dataset. Fine-tuning adapts the model to the specific task (in this case, sentiment classification on the SST-2 dataset).

```python

trainer.train()
```
</br></br></br>
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
</br></br></br>
11. Finally, we want to display the accuracy before and after fine-tuning to quantify the performance improvement.

```python

print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.2f}')
print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.2f}')
```
</br></br></br>
12. Save the file and execute the code to see the fine-tuning happen and the difference before and after. Remember we are only using a very small subset of the dataset, but we are also fine-tuning and testing with the same subset. This will take several minutes to run.

```
python lab3.py
```
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 4 - Adding a model to Hugging Face**

**Purpose: In this lab, we'll push our fine-tuned model to Hugging Face.**
</br></br></br>
1. We're going to take our fine-tuned model from Lab 3 and share it into Hugging Face for general availability. First, we need to save the version of the model (and the tokenizer) that were updated after the process. Edit *lab3.py* and add the two lines below *at the bottom of the file* to save the results in a new local directory.

```python
model.save_pretrained("./my-fine-tuned-model")
tokenizer.save_pretrained("./my-fine-tuned-model")
```
</br></br></br>
2. Now, run the program again with the changes to produce and save the tuned version of the model. Make a note also of the % accuracy *AFTER* the training so we can use it for comparison later. **While this step is running, you can do steps 3 - 5 to get authorization setup!**

```bash
python lab3.py
```
![run to save model](./images/hug63.png?raw=true "Run to save model")
</br></br></br>

3. Switch back to your browser and the huggingface.co site. Make sure you are logged in to your Hugging Face account. We need to have an access token to work with. Go to this URL: https://huggingface.co/settings/tokens/new?tokenType=write to create a new token. (Alternatively, you can go to your user Settings, then select Access Tokens, then Create new token, and select Write for the token type.) Select a name for the token and then Create token.

![creating a new token](./images/hug29.png?raw=true "Creating a new token")
</br></br></br>
4. After you click the Create button, your new token will be displayed on the screen. Make sure to Copy it and save it somewhere you can get to it for the next steps. You will not be able to see it again.

![new token displayed](./images/hug30.png?raw=true "New token displayed")  
</br></br></br>

5. Run the following command to login with your Hugging Face account credentials. (If your job is not finished yet, you can right-click and select in the terminal and select *Split Terminal* to get a second one.) Replace "*<YOUR_SAVED_TOKEN>*" with the actual value of the token you created in the earlier steps.  

```
huggingface-cli login --token <YOUR_SAVED_TOKEN>
```

![logging in with token](./images/hug38.png?raw=true "Logging in with token") 
</br></br></br>

6. After your token is ready **and** the run is complete, take a look at the files in the "./my-fine-tuned-model" directory. You should see a number of files needed to represent the model.

```bash
ls -la ./my-fine-tuned-model
```
![model files](./images/hug65.png?raw=true "Model files")
</br></br></br>

7. Create a new file that will be used to upload the model files (suggested name: *upload-model.py*). Put the following code in it and save it, replacing *<your-account>* with your Hugging Face account name. You can also change the model name from *ft-model-1* to something else if you want.

```bash
code upload-model.py
```

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("./my-fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./my-fine-tuned-model")

# Replace 'your-username/your-model-name' with your Hugging Face username and desired repository name
model.push_to_hub("<your-account>/ft-model-1")
tokenizer.push_to_hub("<your-account>/ft-model-1")
```
</br></br></br>
8. Save the changes to *upload-model.py* and then run it to actually create the repository in your Hugging Face area and upload the files to it. 

```bash
python upload-model.py
```
</br></br></br>
9. Now, you can go to your area on https://huggingface.co/your-account/ft-model-1/tree/main and browse the files to verify that the files were uploaded as expected.

![new repo](./images/hug58.png?raw=true "New repo")
</br></br></br>
10. Click on the "Use this model" button and then the "Transformers" link to see how to use your updated model in code.

![use model](./images/hug59.png?raw=true "Use model")
</br></br></br>
11. Now, let's create some code to download the model from Hugging Face and evaluate it. At the end, we'll print out the accuracy of the model as-is. Create a new file (suggested name *test-model.py*) and then add the code below into it (**substituting in your username/repo in the "tokenizer =" and "model = " lines**).

```bash
code test-model.py
```
```python
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader

train_dataset = load_dataset('glue', 'sst2', split='train[:1%]')
test_dataset = load_dataset('glue', 'sst2', split='validation[:1%]')

tokenizer = AutoTokenizer.from_pretrained("<your account>/ft-model-1")
model = AutoModelForSequenceClassification.from_pretrained("<your account>/ft-model-1")

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

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
print(f'Accuracy of fine-tuned model: {pre_fine_tune_accuracy:.2f}')
```
</br></br></br>

12. Save the file and execute it. At the end, take note of the accuracy value it prints out. It should be the same as the *after training* value from the earlier run. This demonstrates that our fine-tuned model was correctly captured and stored in our new repository.

```bash
python test-model.py
```

![testing model](./images/hug64.png?raw=true "Testing model")
</br></br></br>

(OPTIONAL) If you want, you can select the *README.md* file in the list of files, click on the *edit* link, update any information you want to show up for the model, and then scroll all the way down and commit the changes back to the main branch of the repository.

![adding model info](./images/hug60.png?raw=true "Adding model info")
![adding model info](./images/hug61.png?raw=true "Adding model info")
![adding model info](./images/hug62.png?raw=true "Adding model info")
</br></br></br>

<p align="center">
**[END OF LAB]**
</p>
</br></br>


**Lab 5 - Creating a Sentiment Analysis Web App using Hugging Face and Gradio**

**Purpose: In this lab, we'll create a web-based sentiment analysis application using Hugging Face transformers and Gradio. This app will analyze the sentiment of a given text and classify it as positive, neutral, or negative.**
</br></br></br>
1. Create a new file (suggested name *app.py*) and open it in the editor. Add the imports for the necessary libraries. These imports bring in the model processing and tokenizer from Hugging Face, tools for numerical calculations, and the Gradio library to build the web interface.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import gradio as gr
```
</br></br></br>
2. Now, we need to define the pre-trained model path that we want to use for sentiment analysis. We use one which is specifically trained on Twitter data for classifying sentiments.

```python
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```
</br></br></br>
3. Next is loading the tokenizer from the pre-trained model. This converts the text input into a format the model can understand. It ultimately handles breaking sentences into tokens, padding, and converting tokens into numerical data.

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
```
</br></br></br>
4. Load the model configuration that contains the parameters for the model. This includes elements such as the number of the classes for classification and the architecture details. This is helpful in getting the right settings in place for the model.

```python
config = AutoConfig.from_pretrained(model_path)
```
</br></br></br>
5. And load the pre-trained model that will perform the sentiment classification.
```python
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```
</br></br></br>
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
</br></br></br>
7. We're ready to create the Gradio web interface for the app. This creates the interface, connects the previous function to process the input and displays the sentiment as an output label.

```python
gradio_app = gr.Interface(
    theme=gr.themes.Base(),
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your text here..."),
    outputs="label",
    examples=[
        ["I'm excited about the movie!"],
        ["The weather is great today."],
        ["That meeting was a total waste of time!"],
        ["I'm frustrated with the outcome."],
        ["That was an ok visit."],
        ["I'm could take it or leave it."]
    ],
    title='Sentiment Analysis App',
    description='This app classifies a sentiment as positive, neutral, or negative.'
)
```
</br></br></br>
8. Finally, we'll add code to launch the web app. Save your changes to the file after this.

```python
gradio_app.launch(share=True)
```
</br></br></br>
9. Now, you're ready to test the app. Run the code below to start it. Then you can input sentences to test the sentiment analysis.

```
python app.py
```

10. After this starts running, you should see a line that says "Running on public URL: <URL>". You can go to that address and see the running app. Or you may see a pop-up in the codespace where you can click on the button to get to the app. 

![public address](./images/hug42.png?raw=true "Public address") 

Alternatively, you can go to the PORTS tab in the codespace, find the row for the app, and click on the globe icon to open the application.
![open from codespace](./images/hug43.png?raw=true "Open from codespace") 
</br></br></br>
11. When the app is opened, you can click on one of the pre-populated examples or type your own in and click on *Submit* to see the sentiment result.
![using the app](./images/hug69.png?raw=true "Using the app") 
</br></br></br>
12. When done with the running app, you can use *CTRL+C* back in the terminal to end it.
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Sharing our app to Hugging Face Spaces**

**Purpose - In this lab, we'll share the Gradio app we created in the last lab to a Hugging Face Space**
</br></br></br>
1. Make sure you're signed in to your Hugging Face account. We'll use the Hugging Face CLI again to create a new Space to upload our Gradio app to. Run the command below from the codespace terminal.  (You can stop the Gradio app if you want to use the same terminal or switch to a new one.)

```
huggingface-cli repo create  --type space --space_sdk gradio -y sentiment-analysis-app
```
After this, you'll be able to see a link for the new repo as well as other information. (You can ignore the git-lfs warning.)
![creating the space](./images/hug45.png?raw=true "Creating the space") 

If you want, you can open up that link and look at the empty space on huggingface.co.

2. Now, we're ready to clone down the new space area.
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analysis-app
cd sentiment-analysis-app
```
There's also clone guidance on the website.  
</br></br></br>
3. Create a *requirements.txt* file with these contents and save it. (Hint: *code requirements.txt*, but make sure you're in the cloned directory.)

```
transformers
gradio
scipy
numpy
torch
```
</br></br></br>
4. Copy the *app.py* file from the last lab into the cloned directory. Add it and the *requirements.txt* file to Git and execute the commands from the dialog in your Codespace terminal to get the app put into Spaces.

```bash
cp ../app.py .
git add app.py requirements.txt
git commit -m "Add Gradio sentiment analysis app"
huggingface-cli upload --repo-type space sentiment-analysis-app .
```
</br></br></br>
5. Once pushed, the Hugging Face platform will automatically build and deploy your Gradio app. You can look at the build log on the site and monitor progress. The screenshot below shows the location of the button to look at the build logs for the app as it is building.
![viewing the build log](./images/hug46.png?raw=true "Viewing the build log") 
</br></br></br>
6. When the build process is complete and the ap is deployed, you can switch back to the "App" page to see your Gradio app live on the Hugging Face Space. Test it by entering some text in the input box and see the sentiment classification.
![app running in space](./images/hug47.png?raw=true "App running in space")
</br></br></br>
7. You can share the URL of your space. It's

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
