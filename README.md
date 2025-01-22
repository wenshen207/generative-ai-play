# Generative AI Project

This project demonstrates how to use OpenAI's API to create text embeddings and perform chat-based queries using those embeddings. The embeddings are saved locally in a CSV file to ensure data privacy and prevent data leakage.

## Introduction

This project consists of two main scripts:

1. `openaiEmbed.py`: This script reads text data from a file, splits it into smaller chunks, generates embeddings for each chunk using OpenAI's API, and saves the embeddings to a CSV file.
2. `openaiChat.py`: This script loads the embeddings from the CSV file, generates an embedding for a user query, finds the most similar text chunks based on cosine similarity, and uses OpenAI's chat model to generate a response based on the most similar text chunks.

## Requirements

- Python 3.11
- OpenAI API key

## Installation

1. Install Python 3.11 from the official [Python website](https://www.python.org/downloads/).
2. Install the required Python packages using pip:

    ```sh
    pip install openai numpy
    ```

## Setup

1. Rename the `constants.py` file to `constants.py` and replace `"your key here"` with your actual OpenAI API key.

    ```python
    # constants.py
    APIKEY = "your key here"
    ```

2. Ensure you have a `data.txt` file with the text data you want to embed.

## Usage

### Generating Embeddings

Run the `openaiEmbed.py` script to generate embeddings for the text data and save them to a CSV file.

```sh
python openaiEmbed.py
```

### Chat with Embeddings

Run the `openaiChat.py` script to start a chat session. The script will load the embeddings from the CSV file, generate an embedding for your query, find the most similar text chunks, and use OpenAI's chat model to generate a response.

```sh
python openaiChat.py
```

## Example

1. Add your text data to `data.txt`.
2. Run `openaiEmbed.py` to generate embeddings.
3. Run `openaiChat.py` and enter your query when prompted.

## Project Structure

```
.
├── constants.py
├── data.txt
├── openaiChat.py
├── openaiEmbed.py
├── README.md
├── text_embeddings.csv
```

## License

This project is licensed under the MIT License.