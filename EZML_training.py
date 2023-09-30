import pandas as pd
from ML_main import SensitivityModel
import os, datetime, shutil, time

# Create a new directory to store the files
if not os.path.exists('archive'):
    os.mkdir('archive')

# Generate a unique timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a new directory inside 'archive' with the timestamp as its name
new_dir = os.path.join('archive', timestamp)
os.mkdir(new_dir)

# Load the test data
df = pd.read_excel('training and testing/testing_data.xlsx')

# Save a timestamped copy of the test data in the new folder
test_data_filename = f"{new_dir}/testing_data_{timestamp}.xlsx"
df.to_excel(test_data_filename, index=False)

# Initialize the model
model = SensitivityModel('training and testing/training_data.xlsx')

# Initialize counters for the results
total = 0
successful = 0
failed_predictions = []

# Iterate through each row in the test data
for index, row in df.iterrows():
    text = row['Text']  # assuming the column name is 'Text'
    actual_label = row['Label']  # assuming the column name is 'Label'
    predicted_label = model.predict_sensitivity(text)[0]
    
    # Determine whether the prediction was a pass or fail
    result = 'PASS' if actual_label == predicted_label else 'FAIL'

    # Increment counters
    total += 1
    if result == 'PASS':
        successful += 1
    else:
        # Add failed prediction to the list
        failed_predictions.append((text, actual_label))

    # Print the results to the console
    print(f"Text: {text}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {predicted_label}")
    if predicted_label == 1:
        print(f"This phrase indicates confidentiality")
    else:
        print(f"This phrase does not indicate confidentiality")
    print(f"Result: {result}")
    print("---")

# Calculate success rate
success_rate = successful / total * 100

# Print summary of results
print("\n"*5)
print(f"Total number of tests: {total}")
print(f"Number of successful tests: {successful}")
print(f"Success rate: {success_rate:.2f}%")
print("_________________________________________")
print(f"FAILED PREDICTIONS: {failed_predictions}")

# Create DataFrame from failed predictions and export to Excel with the correct label
if failed_predictions:
    # DataFrame for all failed predictions data
    failed_predictions_df = pd.DataFrame(failed_predictions, columns=['Text', 'Label'])
    
    # Generate a filename in the new folder
    failed_predictions_filename = f"{new_dir}/failed_predictions_{timestamp}.xlsx"
    
    failed_predictions_df.to_excel(failed_predictions_filename, index=False)

# Save a copy of the original training data
shutil.copy('training and testing/training_data.xlsx', f"{new_dir}/training_data_original_{timestamp}.xlsx")


# If there are any failed predictions
if failed_predictions:
    # Load the training data
    training_df = pd.read_excel('training and testing/training_data.xlsx')

    # Append the failed predictions to the training data only if they don't already exist in the training data
    for index, row in failed_predictions_df.iterrows():
        if not ((training_df['Text'] == row['Text']) & (training_df['Label'] == row['Label'])).any():
            # Ask for user confirmation for each failed prediction
            print(f"\n\nDo you want to add the following failed prediction to the training data?\n\nText:\n{row['Text']}\nLabel: {row['Label']}\n\n(yes/no)")
            user_input = input().lower()

            if user_input in ["yes", "y"]:
                print("You have chosen to add this prediction to the training data.")
                print("If you want to change the label before adding, type the new label now. Otherwise, press Enter. If you changed your mind and don't want to add this anymore, type 'cancel'.", end=' ')
                new_label = input().strip()

                # If the user typed something for the new label, use it; otherwise, keep the original label
                if new_label:
                    if new_label.lower() == 'cancel':
                        print("Cancelled adding the failed prediction.")
                        continue
                    else:
                        row['Label'] = new_label
                        print(f"Label changed to: {new_label}")

                print("Adding failed prediction to training data...")
                training_df = training_df._append(row, ignore_index=True)
                time.sleep(2)  # wait 2 seconds
                print("Failed prediction added successfully.")
            else:
                print("Did not add the failed prediction to the training data.")

    # Generate a filename in the new folder
    updated_training_filename = f"{new_dir}/training_data_updated_{timestamp}.xlsx"

    # Export the updated training data to Excel
    training_df.to_excel(updated_training_filename, index=False)

    # Export the updated training data back to original file as well
    print("\n\nDo you want to update the actual training data? (yes/no)")
    user_input = input().lower()

    if user_input in ["yes", "y"]:
        print("Updating training data...")
        training_df.to_excel('training and testing/training_data.xlsx', index=False)
        print("Training data updated successfully.")
    else:
        print("Did not update the training data.")
