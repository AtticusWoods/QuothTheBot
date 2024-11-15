import random

# Specify the number of line pairs to sample
sample_size = 1000

# Read the file once to get the number of lines
with open('trainingdata/modern.txt', 'r') as f:
    total_lines = sum(1 for line in f)

# Generate a set of random line indices to sample from
sample_indices = set(random.sample(range(total_lines), sample_size))

# Open both files and read only the selected lines
modern_texts = []
shakespeare_texts = []

with open('trainingData/modern.txt', 'r') as modern_file, open('trainingData/original.txt', 'r') as shakespeare_file:
    for idx, (modern_line, shakespeare_line) in enumerate(zip(modern_file, shakespeare_file)):
        if idx in sample_indices:
            modern_texts.append(modern_line.strip())
            shakespeare_texts.append(shakespeare_line.strip())

# Now, `modern_texts` and `shakespeare_texts` contain the sampled pairs.
print(modern_texts[:5])  # Displaying the first few samples as a check
print(shakespeare_texts[:5])
