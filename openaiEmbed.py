from openai import OpenAI
import textwrap
import constants
import csv


client = OpenAI(api_key=constants.APIKEY)
# Read the text from a .txt file
file_path = "data.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

print("Text loaded from file:")
print(text[:200])  # Display first 200 characters for a preview


# Split text into smaller chunks (e.g., paragraphs)


max_tokens_per_chunk = 500  # This is a rough limit to avoid exceeding API limits
chunks = textwrap.wrap(text, max_tokens_per_chunk)

print(f"Total number of chunks: {len(chunks)}")



# Create embeddings for each chunk
embeddings = []

for chunk in chunks:
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    # Extract the embedding
    embedding = response.data[0].embedding
    embeddings.append(embedding)

print(f"Generated {len(embeddings)} embeddings.")



csv_file = "text_embeddings.csv"

# Save embeddings to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header (optional)
    writer.writerow(['text_chunk'] + [f'dim_{i}' for i in range(len(embeddings[0]))])
    
    # Write each text chunk and its corresponding embedding
    for chunk, embedding in zip(chunks, embeddings):
        writer.writerow([chunk] + embedding)

print(f"Embeddings saved to {csv_file}")