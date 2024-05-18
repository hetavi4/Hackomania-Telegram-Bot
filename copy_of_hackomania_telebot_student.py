## 🤖 The Telegram Bot

//start a conversation with `@botfather` on telegram, which is the bot used to create and manage other bots.
- Use the command `/newbot` to create a new bot, pick a name and an username for it and botfather will share with you the bot's token!

//That token is very important, since it will be the only way to directly control the bot and reply to messages using the API. That means that the only thing we need to control our bot is to send HTTP requests to telegram servers! Everything that you can do with that token to control the bot is presented in the bot API reference (https://core.telegram.org/bots/api)
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip3 install pyTelegramBotAPI

# @title ⚙️ Configure Telegram Token

#Access your Gemini API key

from google.colab import userdata
import telebot

telegram_bot_token = 'TELEGRAM_TOKEN'  # @param {type: "string"}

try:
  TELEGRAM_API_KEY=userdata.get(telegram_bot_token)
  bot = telebot.TeleBot(TELEGRAM_API_KEY)
  bot.get_me()
except userdata.SecretNotFoundError as e:
   print(f'Secret not found\n\nThis expects you to create a secret named {telegram_bot_token} in Colab\n\nMessage botfather on telegram to create a new bot and get that token\n\nStore that in the secrets section on the left side of the notebook (key icon)\n\nName the secret {telegram_bot_token}')
   raise e
except userdata.NotebookAccessError as e:
  print(f'You need to grant this notebook access to the {telegram_bot_token} secret in order for the notebook to access your Telegram Bot on your behalf.')
  raise e
except Exception as e:
  # unknown error
  print(f"There was an unknown error. Ensure you have a secret {telegram_bot_token} stored in Colab and it's a valid key from telegram")
  raise e

"""For example, to check if new messages for your bot are present, you make a HTTP request on the right endpoint (`getUpdates`).
For that you will use the token that botfather gave you.
Let's try it now! We will use the requests module to make an HTTP GET request to the following url.
`https://api.telegram.org/bot<YOURTOKEN>/getUpdates`
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip3 install requests

import requests

url = f"https://api.telegram.org/bot{userdata.get(telegram_bot_token)}/getUpdates"
response = requests.get(url)

print("Status Code:", response.status_code)
print("Response Content:", response.content)

"""Since your bot has just been created, no new messages are present. Now try to start a new conversation with your bot on telegram. That should automatically send the `/start` command. Let's run the same code again."""

response = requests.get(url)

print("Status Code:", response.status_code)
print("Response Content Text:", response.content)

"""Nice! We are able to get new messages from our bot. There has to be one endpoint that we should be able to use to reply to that message. However, we are lazy, no one want to check by hand for new messages. We can make use a library that the community has already written in Python, to abstract all of these API calls away from us, so that we can focus on building the features of the bot.

## 🐍 Using the `pyTelegramBotAPI` library

Python is this little language you might have heard of if you are into computer science, it started out as a small interpreted programming language, but is now powering the backend of a lot of website, is heavily used for data science thanks to `numpy` and `scikit-learn`, and has become the main language that researchers use for training deep learning, or AI models.
As you probably already know, this notbook can run python code.

Now let us install the package that we need for building our telegram bot. We will need the `pyTelegramBotAPI` package.
We actually already ran that command earlier, because we needed it to verify that the TOKEN that you added to the this notebook secrets was valid
```bash
➜ pip3 install pyTelegramBotAPI
```

And that's it! Not let us create the bot itself.
We will need to import the `telebot` module.
"""

import telebot

BOT_TOKEN = userdata.get(telegram_bot_token
)
bot = telebot.TeleBot(BOT_TOKEN)

"""
Now that we created our bot, we need two things:
1. A function that will be called on each new message
2. A way to get new messages

We will first declare the two following function. Each telegram bot needs to have the `/start` command. The first function will simply send a message  when the users inputs `/start`. The second one is slightly more complex, it will simply reply to each message that is received by the bot with the same message."""

@bot.message_handler(commands=['start'])
def on_start(message):
    bot.send_message(message.chat.id, "Beep, boop, starting bot...")

@bot.message_handler(func=lambda msg: True)
def on_message(message):
	bot.reply_to(message, message.text)

"""
Finally, we need to start the bot by checking periodically for new messages, and executing one of those two functions depending on the contents of the message.

This is automatically done by the `bot.infinity_polling()` function.

Based on the `@bot.message_handler` decorators, it will call the correct function when a new message is received.

Lets put everything together into one `init_bot` function."""

def init_bot():
  BOT_TOKEN = userdata.get(telegram_bot_token)
  bot = telebot.TeleBot(BOT_TOKEN)

  @bot.message_handler(commands=['start'])
  def on_start(message):
      bot.send_message(message.chat.id, "Beep, boop, starting bot...")

  @bot.message_handler(func=lambda msg: True)
  def on_message(message):
    bot.reply_to(message, message.text)

  return bot

"""We can now test our bot! Start your bot by running the next cell, and you can now send messages on telegram to check that everything is working well."""

bot = init_bot()
bot.infinity_polling()

"""<center><img src='https://drive.google.com/uc?id=1xLoNveK1XoddmCOJkJ4VTk4R87tKOs8n' width="400"></center>
🎉

## 🤗 LLMs and HuggingFace

Lets now make the LLM part of the bot.

We have a couple of options, as to what we can do. We could:
- Use OpenAI API, to directly access GPT3.5 or GPT4, but every token would cost us money (for your project it might be good to use OpenAI)
- Use a model hosted locally, directly on your computer, but even for smaller models, the weights will not necessarily fit on a computer with limited ram
- Use the free 🤗 Hugging Face [Inference API](https://huggingface.co/inference-api), that allow us to play with OpenSource language models hosted on the platform

We will go for option 3. If you have ever heard of GitHub, HuggingFace is supposed to be the GitHub of AI, where you can find models architectures, model weights and datasets. If everything does not make sense to you that's okay, we are only going to use those models for inference, so we do not need to know every detail.

The model we will use is Mistral7B Instruct https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2.
Its a very popular model that was trained by the company MistralAI, has 7 Billion parameters, and has been finetuned to be used as a chat assistant model.

In order to use the free HuggingFace inference API, you need a HuggingFace account, and once again, an API token to be able to use their API.
To get one go to `Settings > Access Tokens > New Token > Role = write`.
Then copy the generated token, we will use it in the python code that we will be writing.
"""

# @title ⚙️ Configure Hugging Face Token

from google.colab import userdata

hugging_face_token_secret = 'HF_TOKEN'  # @param {type: "string"}

try:
  TELEGRAM_API_KEY=userdata.get(hugging_face_token_secret)
except userdata.SecretNotFoundError as e:
   print(f'Secret not found\n\nThis expects you to create a secret named {hugging_face_token_secret} in Colab\n\nGot to that url and create a write token (https://huggingface.co/settings/tokens)\n\nStore that in the secrets section on the left side of the notebook (key icon)\n\nName the secret {hugging_face_token_secret}')
   raise e
except userdata.NotebookAccessError as e:
  print(f'You need to grant this notebook access to the {hugging_face_token_secret} secret in order for the notebook to access your Telegram Bot on your behalf.')
  raise e
except Exception as e:
  # unknown error
  print(f"There was an unknown error. Ensure you have a secret {hugging_face_token_secret} stored in Colab and it's a valid key from telegram")
  raise e

"""We will not be using the HuggingFace inference API directly, for that we will make use of LangChain (https://www.langchain.com), which is a very popular framework for using and interfacing LLMs with other things in Python (more on that later).
We will also need the transformers library from huggingface, for things like saving chat history.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip3 install langchain transformers

"""### Creating the LLM class

First add those imports, and that line to retrieve your hugging face token at the start of the script.
"""

import os
from langchain_community.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get(hugging_face_token_secret)

"""
We will be using a class for the LLM, so that all the information related to it is stored in that class. The first step is to initialize the [model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). As stated previously, we are using the HuggingFace inference API to run our model."""

class LLM:
  def __init__(self):
    model_string = "mistralai/Mistral-7B-Instruct-v0.2"
    self.chat = []
    self.llm = HuggingFaceHub(repo_id=model_string, model_kwargs={"temperature": 0.5, "max_length":64,"max_new_tokens":512})

  def get_reply(self, instruction):
    return self.llm.invoke(instruction)

"""The first function is called on instantiation of the class, and creates the llm by using HuggingFaceHub, with the model we decided to use for inference.

We can now instantiate the class, add the following line below your class declaration.
"""

llm = LLM()

"""Now, instead of replying back with the same message that got sent to us when we receive a new message, we can send it to the LLM, which should give us a reply to our question! For that we only have to change the following line."""

def init_bot():
  bot = telebot.TeleBot(BOT_TOKEN)

  @bot.message_handler(commands=['start'])
  def on_start(message):
      bot.send_message(message.chat.id, "Beep, boop, starting bot...")

  @bot.message_handler(func=lambda msg: True)
  def on_message(message):
      print(f"Message received! {message}")
      reply = llm.get_reply(message.text)
      bot.reply_to(message, reply)

  return bot

"""Let's test it!"""

bot = init_bot()
bot.infinity_polling()

"""<center><img src='https://drive.google.com/uc?id=1khyxnxGZu_wLi_uo0y4XS10BR7Jixlca' width="400"></center>

Seems to work, except that the model added a question mark, since I forgot to have it in my message. This is a good reminder that, at the end of the day, large language models only job is to predict what would be the next words, given an initial prompt. In order to circumvent that, we need to change our input to the model, our prompt, to make use of the same format that was used when finetuning the model.

Another problem is that the model does not seem to have any memory, if I send a message referencing the previous question, the output is not what I want.

<center><img src='https://drive.google.com/uc?id=1O8QFMWrMWIDcTg9cZxk9S38QYxce96k3' width="400"></center>



Thankfully, we can simply use the 🤗 `transformers` library, which contains all the functions that we need.

### Making it handle conversations

In order to really be able to chat with model, we need to store the chat history.
That way, we will be able to refer to messages previously sent in the chat.
When creating the class, you can see that I already created an empty list `self.chat = []`.
This list will contain all the questions and replies that have been sent and received during the chat.

Another thing we need to do is to tokenize that chat history. This means transforming it into a format that the model understands.

To do that, we will use the `transformers` library.
The format for that list is quite simple.

```python
[
   {"role" : "user", "content" : "Initial question"},
   {"role" : "assistant", "content" : "Reply"},
   {"role" : "user", "content" : "New question"},
   {"role" : "assistant", "content" : "Reply 2"},
   ...
]
```

So now, what we need is to save every message that is sent by the user, as well as every response from the model into that list.

We can simply modify the `get_reply` function.
"""

class LLM:
  def __init__(self):
    model_string = "mistralai/Mistral-7B-Instruct-v0.2"
    self.chat = []
    self.llm = HuggingFaceHub(repo_id=model_string, model_kwargs={"temperature": 0.5, "max_length":64,"max_new_tokens":512})

  def get_reply(self, instruction):
    self.chat.append({"role" : "user", "content" : instruction})

    reply = self.llm.invoke(instruction)
    self.chat.append({"role" : "assistant", "content" : reply})
    return reply

"""Great, we are now saving the chat history with each new message that we receive. The next step is to use that chat history for the prompt that is sent to the llm.

> Note: This bot only supports one conversation with one message being sent at a time.

For that we will have to use `AutoTokenizer` from the `transformers` library.
"""

from transformers import AutoTokenizer

class LLM:
  def __init__(self):
    model_string = "mistralai/Mistral-7B-Instruct-v0.2"
    self.chat = []
    self.llm = HuggingFaceHub(repo_id=model_string, model_kwargs={"temperature": 0.5, "max_length":64,"max_new_tokens":512})
    self.tokenizer = AutoTokenizer.from_pretrained(model_string)

  def get_reply(self, instruction):
    self.chat.append({"role" : "user", "content" : instruction})

    prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False)
    print(prompt)
    reply = self.llm.invoke(prompt)[len(prompt):]
    print(reply)
    self.chat.append({"role" : "assistant", "content" : reply})
    return reply

llm = LLM()

"""Now, our bot should be a able to recall previous messages, since we are sending the whole conversation to the LLM each time we ask a question.

Lets also create a short command that creates a new chat. For that we simply have to make the chat history list empty.
"""

def init_bot():
  bot = telebot.TeleBot(BOT_TOKEN)

  @bot.message_handler(commands=['start'])
  def on_start(message):
      bot.send_message(message.chat.id, "Beep, boop, starting bot...")

  @bot.message_handler(commands=['newchat'])
  def on_new_chat(message):
    llm.chat = []
    bot.reply_to(message, "Starting new chat!")

  @bot.message_handler(func=lambda msg: True)
  def on_message(message):
      print(f"Message received! {message}")
      reply = llm.get_reply(message.text)
      print(message.text)
      print(reply)
      bot.reply_to(message, reply)

  return bot

"""Now lets try!"""

bot = init_bot()
llm.chat = []
bot.infinity_polling()

"""<center><img src='https://drive.google.com/uc?id=19jDtL32UVmF1TEU1SNSsGQwvmMf5EpMf' width="400"></center>

## 💽 Retrieval Augmented Generation

Now what happens if we ask a question to our bot, that is not present in its training database?
For example, something related to recent news. Or a specific topic where the model might hallucinate answers. You can try it!
"""

bot = init_bot()
llm.chat = []
bot.infinity_polling()

"""<center><img src='https://drive.google.com/uc?id=17IXh6uYom94qVG5a17EmZHNpNgPxIW-h' width="400"></center>

As you can see, the training data for this model probably ends around 2021. It does not have accurate information about that topic.
It would be great if we can provide the model with context, or additional information given that prompt.

Retrieval Augmented Generation (RAG) is a technique that is used for that goal.
<center>
<img src='https://media.licdn.com/dms/image/D4D12AQHY76w85U8W5g/article-cover_image-shrink_720_1280/0/1695787886133?e=1710979200&v=beta&t=sV6_ZlY78y55Vnvg9Wfs7dUunL-SwDAFBRcQ14SjqYY' width=1000></center>

Using RAG, we can provide the model with our own documents and data, and the llm will be able to search through our data for us!

In order to use this technique with our bot, we will make use of several of `langchain` modules.
"""

!pip3 install faiss-cpu

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

"""### Using the web

In order to get the right response to our question, we must provide context, for example by helping the model with content from an article talking about the current UK prime minister, Rishi Sunak.

We will load that webpage, extract its contents and chunk it into parts, each part containing some of the webpage content.
All these 'chunks' will be fed into a database, that will only return the relevant parts to our question.

Let us first load the webpage and process it into chunks.
Find the url of the wikipedia page about Rishi Sunak, and paste it below.
"""

# wikipedia_rishi_sunak_url = FIXME
prime_ministers_loader = WebBaseLoader(wikipedia_rishi_sunak_url).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50, length_function = len)

prime_ministers_chunks = text_splitter.transform_documents(prime_ministers_loader)

"""We can then initialize our database using those chunks of data. This database will not contain the chunks directly, but their embeddings, which is a higher order representation of the data, used to group similar content together.

This embedding is computed from the data, using another machine learning model. In this tutorial we are using [sentence-transformers/all-MiniLM-l6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), also available on 🤗 Hugging face.

The database that we will be using is [FAISS](https://github.com/facebookresearch/faiss).

Lets initialize everything, using the data we retrieved from the web.
"""

store = LocalFileStore("./cache")
core_embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key = userdata.get(hugging_face_token_secret),
    model_name="sentence-transformers/all-MiniLM-l6-v2")

embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store)
vectorstore = FAISS.from_documents(prime_ministers_chunks, embedder)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})

"""We can then search from this database, and we will get relevant part of the webpage to add to our llm prompt."""

retriever.get_relevant_documents("who is the current uk prime minister")

"""🎉 Great! We now have to create the function that will, given a list of documents, generate a new query. This new prompt will have the following format:


> Answer the following question, making use of the documents provided below if they are relevant. Do not use those documents and do not mention them if you deem that they do not contain any relevant information. Do not mention the id of the documents. If a document is relevant, add the source of the information by adding a link to the exact url that was used. For that use the 'source' field of the relevant document.
>
> Question: [QUESTION]
>
> Documents:
> - Document 1 [CONTENTS]
- Document 2 [CONTENTS]
- ...

"""

def append_documents_to_instruction(instruction):
    instruction_with_documents = f'''Answer the following question, making use of the documents provided below if they are relevant. Do not use those documents and do not mention them if you deem that they do not contain any relevant information. Do not mention the id of the documents. If a document is relevant, add the source of the information by adding a link to the exact url that was used. For that use the 'source' field of the relevant document.
Question: {instruction}

    '''

    # docs = FIXME How to get documents from the instruction?

    if len(docs) == 0: # If there are no relevant documents, just return the original instruction
        return instruction

    instruction_with_documents += "Documents:\n"

    for i, doc in enumerate(docs):
        instruction_with_documents += f'''- {doc.metadata}
            Content: {doc.page_content}
'''
    return instruction_with_documents

"""We now have to modify our LLM class a little, to use that new function before getting a reply."""

class LLM:
  def __init__(self):
    model_string = "mistralai/Mistral-7B-Instruct-v0.2"
    self.chat = []
    self.llm = HuggingFaceHub(repo_id=model_string, model_kwargs={"temperature": 0.5, "max_length":64,"max_new_tokens":512})
    self.tokenizer = AutoTokenizer.from_pretrained(model_string)

  def get_reply(self, instruction):
    # instruction_with_context = FIXME
    self.chat.append({"role" : "user", "content" : instruction_with_context})

    prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False)
    reply = self.llm.invoke(prompt)
    self.chat.append({"role" : "assistant", "content" : reply})
    return reply

llm = LLM()

"""Lets check if it works!"""

bot = init_bot()
llm.chat = []
bot.infinity_polling()

"""<center><img src='https://drive.google.com/uc?id=1kKviHTfUg5KMplQ7Qmvbcl2VjME_Aff6' width="400"></center>

Great! Our bot can now use content from the web to check answers, or get more up to date information! Now what if we want to get information from a PDF file, for example a research paper?

### Using your own pdf files

We first need to install the pypdf library, we need that library for processing PDF files.
"""

!pip3 install pypdf

"""Since we only communicate with the bot API, we will not directly receive the file when an user sends it. We need to manually download the file, after asking telegram to provide us with the url address.
More info [here](https://core.telegram.org/api/files).

Let's put all that logic into a `dl_file` function, that takes a given message as input. It will use the `requests` library.
"""

def dl_file(message):
    import requests
    file_info = bot.get_file(message.document.file_id)

    download_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
    response = requests.get(download_url)

    if response.status_code == 200:
        with open(message.document.file_name, 'wb') as file:
            file.write(response.content)
        return True
    else:
        return False

"""We now have to add a new handler to our bot, that will check wether or not the document that has been sent is a pdf or not, and will process the file and add it to our database, so that we can search from it.
Lets call that handler `on_document`.
"""

def init_bot():
  bot = telebot.TeleBot(BOT_TOKEN)

  @bot.message_handler(commands=['start'])
  def on_start(message):
      bot.send_message(message.chat.id, "Beep, boop, starting bot...")

  @bot.message_handler(commands=['newchat'])
  def on_new_chat(message):
    llm.chat = []
    bot.reply_to(message,  "Starting new chat!")

  @bot.message_handler(content_types=['document'])
  def on_document(message):
    if message.document.mime_type == 'application/pdf':
        reply = bot.reply_to(message, "⬇️ Downloading file ⬇️")

        # if not FIXME:
            bot.edit_message_text("❌ Failed to download file", reply.chat.id, reply.message_id)
            return

        bot.edit_message_text("🗃️ Adding file to database 🗃️", reply.chat.id, reply.message_id)

        loader = PyPDFLoader(message.document.file_name)
        pages = loader.load_and_split()
        chunks = text_splitter.transform_documents(pages)

        vectorstore.add_documents(chunks)

        bot.edit_message_text("✅ PDF received and added to database", reply.chat.id, reply.message_id)
    else:
        bot.reply_to(message, "For the moment, I only support indexing PDF files. Please send a PDF file.")

  @bot.message_handler(func=lambda msg: True)
  def on_message(message):
      print(f"Message received! {message}")
      reply = llm.get_reply(message.text)
      bot.reply_to(message, reply)

  return bot

"""We can now try sending our own documents to the bot, and ask for information about it! Let's try!"""

bot = init_bot()
llm.chat = []
bot.infinity_polling()

"""## Additional Ressources

How does document loading works ([source](https://python.langchain.com/docs/use_cases/question_answering/))

<center><img src='https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png' width="1000"></center>
"""

