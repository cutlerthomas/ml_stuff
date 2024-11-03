import csv

def process_data(filename):
    # Initial values to hold parameters and results
    alpha, gamma, epsilon, learn, score = "", "", "", "", ""

    # Open CSV to write data
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alpha', 'gamma', 'epsilon', 'learn', 'score'])

        # Read input data
        with open(filename, 'r') as data:
            for line in data:
                line = line.strip()
                if line.startswith("Running"):
                    parts = line.split()
                    alpha = parts[4].rstrip(',')
                    gamma = parts[5].rstrip(',')
                    epsilon = parts[7]
                elif line.startswith("Learn"):
                    # Extract the numeric reward value
                    learn = line.split()[-1]
                elif line.startswith("Score"):
                    # Extract the numeric reward value
                    score = line.split()[-1]

                # Check if all data is collected for one set of parameters and results
                if alpha and gamma and epsilon and learn and score:
                    writer.writerow([alpha, gamma, epsilon, learn, score])
                    # Reset after writing
                    alpha, gamma, epsilon, learn, score = "", "", "", "", ""

    entries = []
    avg_score = 0
    count = 0
    with open('data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['score'] = float(row['score'])  # Convert score to float for sorting
            entries.append(row)
            avg_score += row['score']
            count += 1
    
    # Sort entries by score
    entries_sorted = sorted(entries, key=lambda x: x['score'])
    
    # Get the lowest and highest 5 scores
    lowest_five = entries_sorted[:5]
    highest_five = entries_sorted[-5:]

    print("Avg score: ")
    print(avg_score / count)

    print("Top 5 highest scores:")
    for entry in reversed(highest_five):
        print(entry)

    print("Top 5 lowest scores:")
    for entry in lowest_five:
        print(entry)

if __name__ == '__main__':
    process_data('data.txt')
