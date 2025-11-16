sentences = ["I enjoyed the movie",
             "I did not like the food",
             "He wasn't happy with your request",
             "What a lovely day to be alive"]

word_counts = {}
words_to_ignore = {"i", "the"}

for sentence in sentences:
    words = sentence.split()   # Split into words
    for word in words:
        word = word.lower()    # Normalize the words in lowercase for consistency

        if word in words_to_ignore:
            continue

        word_counts[word] = word_counts.get(word, 0) + 1

# Find the highest count
max_count = max(word_counts.values())

# Find all words with that highest count
most_frequent_words = [word for word, count in word_counts.items() if count == max_count]

print("Word_count:", word_counts)
print("Most frequent count:", max_count)
print("Most frequent words:", most_frequent_words)
