from openai import OpenAI
import csv
import constants
import sys

import numpy as np

csv_file = "text_embeddings.csv"
client = OpenAI(api_key=constants.APIKEY)


# Load embeddings from CSV
loaded_texts = []
loaded_embeddings = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        text = row[0]
        embedding = list(map(float, row[1:]))  # Convert each dimension back to float
        loaded_texts.append(text)
        loaded_embeddings.append(embedding)

# print(f"Loaded {len(loaded_embeddings)} embeddings from {csv_file}")



# Define a function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Generate a query embedding

def call_chat(query):

    query_embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )

    embedding_response = query_embedding.data[0].embedding

    # Find the most similar text based on cosine similarity
    best_match1 = None
    best_match2 = None
    best_match3 = None
    best_score = -1

    for text, embedding in zip(loaded_texts, loaded_embeddings):
        score = cosine_similarity(embedding_response, embedding)
        if score > best_score:
            best_score = score
            best_match3 = best_match2
            best_match2 = best_match1
            best_match1 = text


    # print(f"Best match for the query '{query}' is: '{best_match1}' with score {best_score}")
    # print(f"Best match for the query '{query}' is: '{best_match2}' with score {best_score}")
    # print(f"Best match for the query '{query}' is: '{best_match3}' with score {best_score}")

    match_data = best_match1

    if best_match2 != None:
        match_data = match_data + " " + best_match2

    if best_match3 != None:
        match_data = match_data + " " + best_match3


    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Change your model here
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}, # Change your prompt here
            {"role": "user", "content": f"Answer this query based on the following context:\n\n{match_data}\n\nQuery: {query}"}
        ]
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content

ask = None
while True:
  if not ask:
    ask = input("Prompt: ")
  if ask in ['quit', 'q', 'exit']:
    sys.exit()
  result = call_chat(query=ask)
  print("Anser: ", result)

  ask = None