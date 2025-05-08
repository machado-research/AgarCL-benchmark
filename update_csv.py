import csv

def filter_csv(file_path, threshold):
    # Read the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Keep the first row (header) and filter the rest based on the second column
    header = rows[0]
    filtered_rows = [header] + [row for row in rows[1:] if float(row[0]) <= threshold]
    # Write the filtered rows back to the file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_rows)

# Example usage
csv_file_path = '/home/mamoham3/Results/BEST_DQN_mode_full_game/48d75f26997f6d15791dacbfab75bf7a9c8934fa-f1a040b9-dd773a6b/episodic_rewards.csv'  # Replace with your CSV file path
threshold_value = 6000000  # Replace with the number you want to compare
filter_csv(csv_file_path, threshold_value)