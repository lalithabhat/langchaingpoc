from dotenv import load_dotenv
import os
import pandas as pd

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")

"""## Loaders  
To use data with an LLM, documents must first be loaded into a vector database.
The first step is to load them into memory via a loader
"""
def vectoriseTrainingData():
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.document_loaders.csv_loader import CSVLoader

    # loader = TextLoader("C:/Users/Lalitha Bhat/OneDrive/code/Langchain-Full-Course/RCA Training Data/Incident Training.csv")
    # docs = loader.load()
    loader = CSVLoader(file_path='./RCA Training Data/Incidents Training.csv',
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                        'fieldnames': ['Event Description', 'Cause Analysis', 'Corrective Actions' , 'Corrective Actions Description']
                        })
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

    with open("rca_vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    

"""## Prompts
With an LLM you have the possibility to give it an identity before a conversation or to define how question and answer should look like.
"""
def searchServices(issueDetail):
    """## Loading the database
    Before using the database, it must of course be loaded again.
    """
    import pickle
    with open("./rca_vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    from langchain.prompts import PromptTemplate
    prompt_template = getPromptOfferingSearch();
    # PROMPT = PromptTemplate(
    #     template=prompt_template
    # )
    PROMPT = PromptTemplate.from_template(prompt_template)

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

    # result = qa({"question": leadquery}, return_only_outputs=True)
    result = qa(issueDetail , return_only_outputs=True)
    answer = result["result"]
    source = result["source_documents"]
    print(answer)
    print(source)
    return pd.DataFrame({
        "Cause" : ["Cause Analysis", "Source"],
        "Value" : [answer, source]
    })

def getPromptOfferingSearch():
    return '''You are a helpful analyst that has been trained in commonly occuring issues in the ship and its root cause and preventive measures.Based on the issue description, identify the root cause for new issues.
    Answer only from the fact shared with you, don't make up responses.Say I don't know when you don't know the answer.
    {context}

    Question: {question}
    Answer here:'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    issueForm = '''
       "On November 16th 2022, hot well was contaminated by fuel due to a leak from Fuel Oil purifer # 1 heater. 

    HFO purifier and heater were isolated. Faulty steam heater has been replaced. Upon investigation a new model of steam heater has been ordered with requisition ED035682 and cleaning of hot well begun on the same day. 
    HFO reached both boilers as confirmed by boiler water sample analysis done by shoreside lab. All boilers' blow down operation were suspended.

    It has been necessary to drain and pressure wash boiler # 1 and Eco # 1-2-3 in order to be able to resume boiler blow down and maintain water chemistry in range.

    Due to welding repairs inside furnace of boiler # 2 and imminent full repairs ( due in February 2023) it has been decided not to drain Boiler # 2.  Boiler # 2 blow down operation are done only internally and water chemistry is monitored closely. 

    Upon investigation and consultation with maker of purifier a new model of steam heater has been ordered with requisition ED035682

    See attached correspondence for further details."

    '''
    # vectoriseTrainingData()
    result = searchServices(issueForm)

    print(result)