import openai
import chromadb
import cohere
import os
from chromadb.utils import embedding_functions


openai.api_key = os.environ["OPENAI_API_KEY"]
co = cohere.Client('zSGn7I2UykHP9vYylMqU4j9giwo6dlKgdka5hCQF')
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key= openai.api_key,
                model_name="text-embedding-ada-002"
            )
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")


openai.api_key = os.environ["OPENAI_API_KEY"]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key= openai.api_key,
                model_name="text-embedding-ada-002")

# Required Libraries
import pandas as pd
# Step 1: Load the CSV in Colab
# This assumes you've uploaded the CSV file in Colab and it's named 'adventistFAQ.csv'
df = pd.read_csv('test.csv')

text_list = df.iloc[:, 0].dropna().tolist()
#documents = [Document(text=t) for t in text_list]
ids = [str(i) for i in range(len(text_list))]

collection.add(documents=text_list, ids=ids)

def get_reranked_documents(query, n_query=8, n_rerank=3):
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=n_query
    )
    documents_from_results = results['documents'][0]

    # Rerank the results
    reranked_results = co.rerank(query=query, documents=documents_from_results, top_n=n_rerank, model='rerank-multilingual-v2.0')

    # Format the reranked results
    formatted_documents = []
    for idx, result in enumerate(reranked_results, start=1):
        text = result.document['text']
        formatted_documents.append(f"資料{idx}：{text}")

    # Join the formatted documents into a single string for display
    output = "參考資料：\n\n" + "\n\n".join(formatted_documents)
    return output


cs_bot_system_message = {
    "role": "system",
    "content": "你是港安醫療中心的客服，請禮貌地在Whatsapp上協助客戶的問題。"
}


# Modify the client_response function to get human input
def client_response(messages):
    # Print the last message sent to the client
    print(f"Sending to Client: {messages[-1]['content']}")

    # Get input from the user (real human) as the client's response
    human_input = input("Enter Client's Response: ")

    return human_input


def cs_bot_response(query):
    # Fetch the reranked documents based on the client's message
    reference = get_reranked_documents(query=query)
    print("相關參考資料", reference)

    # Add the reference to the system message content
    system_message_content = f'''
    你是港安醫療中心的客服，請禮貌地在Whatsapp上協助客戶的問題。
    你在參考資料中可以找到港安醫療中心的相關資料。

    你是在whatsapp 上回答客戶，所以請盡量保持簡短，每次回覆盡量不多于20個字。

    {reference}
    '''

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_message_content}, {"role": "user", "content": query}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['message']['content']


# def converse(client_start, num_exchanges=5):
#     cs_bot_conversation = []

#     client_message = {
#         "role": "user",
#         "content": client_start
#     }
#     cs_bot_conversation.append(client_message)

#     for _ in range(num_exchanges):
#         cs_bot_content = cs_bot_response(cs_bot_conversation)

#         cs_bot_reply = {
#             "role": "assistant",
#             "content": cs_bot_content
#         }
#         cs_bot_conversation.append(cs_bot_reply)

#         client_content = client_response(cs_bot_conversation)

#         client_reply = {
#             "role": "user",
#             "content": client_content
#         }
#         cs_bot_conversation.append(client_reply)

#     return cs_bot_conversation