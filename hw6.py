import requests
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def fetch_text(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Mapping function: splits text into words and assigns a count of 1 to each
def map_function(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [(word, 1) for word in words]

def shuffle_function(mapped_values):
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()

def reduce_function(shuffled_values):
    reduced = {}
    for key, values in shuffled_values:
        reduced[key] = sum(values)
    return reduced

# Parallelized MapReduce function
def parallel_map_reduce(text, num_threads=4):
    words = re.findall(r'\b\w+\b', text.lower())

    chunk_size = len(words) // num_threads
    chunks = [words[i * chunk_size:(i + 1) * chunk_size] for i in range(num_threads)]
    if len(words) % num_threads:
        chunks.append(words[num_threads * chunk_size:])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        mapped_results = list(executor.map(map_function, [' '.join(chunk) for chunk in chunks]))

    combined_mapped_values = [item for sublist in mapped_results for item in sublist]

    shuffled_values = shuffle_function(combined_mapped_values)
    reduced_values = reduce_function(shuffled_values)

    return reduced_values

def visualize_top_words(word_counts, top_n=10):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*sorted_words)

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color="skyblue")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Top 10 Most Frequent Words")
    plt.show()

if __name__ == '__main__':
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt" 
    
    text = fetch_text(url)

    word_counts = parallel_map_reduce(text, num_threads=4)

    visualize_top_words(word_counts)