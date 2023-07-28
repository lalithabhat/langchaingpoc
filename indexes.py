from dotenv import load_dotenv
import os
import pandas as pd
import tiktoken

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
# API_KEY = os.environ.get("API_KEY")
API_KEY = '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'

"""## Loaders  
To use data with an LLM, documents must first be loaded into a vector database.
The first step is to load them into memory via a loader
"""

from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
docs = loader.load()

"""## Text splitter
Texts are not loaded 1:1 into the database, but in pieces, so called "chunks". You can define the chunk size and the overlap between the chunks.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

documents = text_splitter.split_documents(docs)
documents[0]

"""## Embeddings
Texts are not stored as text in the database, but as vector representations.
Embeddings are a type of word representation that represents the semantic meaning of words in a vector space.
"""

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

"""## Loading Vectors into VectorDB (FAISS)
As created by OpenAIEmbeddings vectors can now be stored in the database. The DB can be stored as .pkl file
"""

from langchain.vectorstores.faiss import FAISS
import pickle

vectorstore = FAISS.from_documents(documents, embeddings)

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

"""## Loading the database
Before using the database, it must of course be loaded again.
"""

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

"""## Prompts
With an LLM you have the possibility to give it an identity before a conversation or to define how question and answer should look like.
"""
def searchServices(leadquery):
    from langchain.prompts import PromptTemplate
    prompt_template = getPromptOfferingSearch();
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    """## Chains
    With chain classes you can easily influence the behavior of the LLM
    """

    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA

    chain_type_kwargs = {"prompt": PROMPT}

    llm = OpenAI(openai_api_key=API_KEY,verbose=True)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents = True
    )

    result = qa(leadquery, return_only_outputs=True)
    answer = result["result"]
    source = result["source_documents"]
    # print(answer)
    # print(source)
    return pd.DataFrame({
        "Offering" : ["Service Offerings", "Source"],
        "Value" : [answer, source]
    })

def getPromptOfferingSearch():
    return '''You are a helpful search tool that helps identify the right service offering based on the customer's query.
    Answer only from the fact shared with you, don't make up responses. Say I don't know when you don't know the answer.
    {context}

    Question: {question}
    Answer here:'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    leadForm = '''
        Will I get my clothes by Tuesday?
    '''
    result = searchServices(leadForm)

    print(result)