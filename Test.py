import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = "..." # Call your model here
model.load_state_dict(torch.load("....."))  # Load Pretrained model
model.to(device)
model.eval()

# Define data transformations for the test image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Path to the folder containing test data
rootFolderPath = "path/to/test-data"

# Function to classify an image as fake or real based on a threshold
def fake_or_real(file_path, threshold):
    """
    Classify an image as fake or real based on a threshold.

    Args:
        file_path (str): Path to the image file.
        threshold (float): Threshold value for classification.

    Returns:
        str: "fake" if the image is classified as fake, "real" otherwise.
    """
    # Load the test image
    test_image = Image.open(file_path).convert("RGB")
    test_image = transform(test_image).unsqueeze(0).to(device)

    # Pass the test image through the model
    with torch.no_grad():
        output = model(test_image)

    # Calculate the difference between the input and output images
    mse_loss = nn.MSELoss(reduction="mean")
    loss = mse_loss(output, test_image)

    # Classify the image based on the loss value
    if loss.item() >= threshold:
        return "fake"
    else:
        return "real"

# Iterate over real and fake files, and store them in dictionaries
fake_dict = {}
real_dict = {}
for root, dirs, files in os.walk(rootFolderPath):
    for filename in files:
        if filename.endswith(tuple(['.png', '.jpg', '.jpeg'])):
            file_path = os.path.join(root, filename)
            if 'fake' in file_path:
                if root in fake_dict.keys():
                    fake_dict[root].append(file_path)
                else:
                    fake_dict[root] = [file_path]
            else:
                if root in real_dict.keys():
                    real_dict[root].append(file_path)
                else:
                    real_dict[root] = [file_path]

# Calculate TPR and FPR for different thresholds
thresholds = np.linspace(0, 1, num=100)
tprs = []
fprs = []

for threshold in thresholds:
    # Initialize counters
    real_count = 0
    fake_count = 0
    total_real = 0
    total_fake = 0

    # Iterate over fake files and classify them
    for folder_path in fake_dict.keys():
        local_count = 0
        total_local_count = len(fake_dict[folder_path])
        for file_path in fake_dict[folder_path]:
            total_local_count += 1
            result = fake_or_real(file_path, threshold)
            if result == "fake":
                fake_count += 1
                local_count += 1
        print("In the folder {0}, which was fake, our accuracy was {1:.2f}%.".format(folder_path, (local_count / total_local_count) * 100))

    # Iterate over real files and classify them
    for folder_path in real_dict.keys():
        local_count = 0
        total_local_count = len(real_dict[folder_path])
        for file_path in real_dict[folder_path]:
            total_local_count += 1
            result = fake_or_real(file_path, threshold)
            if result == "real":
                real_count += 1
                local_count += 1
        print("In the folder {0}, which was real, our accuracy was {1:.2f}%.".format(folder_path, (local_count / total_local_count) * 100))

    # Calculate accuracy
    real_accuracy = (real_count / total_real) * 100
    fake_accuracy = (fake_count / total_fake) * 100

    # Print the results
    print(f"Accuracy on testing real data is: {real_accuracy:.2f}%", 'total real images are', total_real)
    print(f"Accuracy on testing fake data is: {fake_accuracy:.2f}%", 'total fake images are', total_fake)

    # Initialize counters for true positives, false positives, true negatives, and false negatives
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Classify fake images and calculate true positives and false positives
    for folder_path in fake_dict.keys():
        for file_path in fake_dict[folder_path]:
            result = fake_or_real(file_path, threshold)
            if result == "fake":
                true_negatives += 1
            else:
                false_positives += 1

    # Classify real images and calculate true positives and false negatives
    for folder_path in real_dict.keys():
        for file_path in real_dict[folder_path]:
            result = fake_or_real(file_path, threshold)
            if result == "real":
                true_positives += 1
            else:
                false_negatives += 1

    # Calculate true positive rate (TPR) and false positive rate (FPR)
    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)

    # Print the results
    print("Threshold:", "{:.4f}".format(threshold))
    print("TPR:", tpr)
    print("FPR:", fpr)

    # Append TPR and FPR to the lists
    tprs.append(tpr)
    fprs.append(fpr)

# Plot the ROC curve
plt.figure()
plt.plot(fprs, tprs, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve-CBAM')
plt.legend(loc="lower right")
plt.show()
