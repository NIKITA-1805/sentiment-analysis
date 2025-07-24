import csv

input_files = ['train.txt', 'test.txt']  # Replace with your actual file names if different
output_file = 'reviews.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['review', 'sentiment'])  # CSV header

    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                # Split label and text
                label, text = line.split(' ', 1)
                if label == '__label__2':
                    sentiment = 'positive'
                elif label == '__label__1':
                    sentiment = 'negative'
                else:
                    # Skip any malformed lines
                    continue
                writer.writerow([text, sentiment])

print('Done! Output written to reviews.csv')
