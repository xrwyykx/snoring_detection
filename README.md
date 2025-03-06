# Snore_Detection_project
This project is focused on detecting snoring sounds from audio recordings using machine learning techniques. The dataset includes labeled audio samples of snoring and non-snoring sounds, and the project involves preprocessing the audio data, extracting features, training a machine learning model, and evaluating its performance.

# Project Structure
The project is structured as follows:
```
├── data/
│   ├── 0/  # Folder containing non-snoring audio files
│   ├── 1/  # Folder containing snoring audio files
│   ├── train/  # Training data (created by the script)
│   │   ├── snoring/
│   │   └── non-snoring/
│   └── test/  # Testing data (created by the script)
│       ├── snoring/
│       └── non-snoring/
├── models/  # Saved models
├── inference_audios # For storing test audios
├── src/
|   ├──logs 
│   ├── preprocess.py  # Data preprocessing script
│   ├── feature_extraction.py  # Feature extraction script
│   ├── train.py  # Model training script
│   ├── evaluate.py  # Model evaluation script
│   ├── utils.py  # Utility functions
│   ├── config.py  # Configuration settings
│   ├── model.py  # Model architecture
│   └── split_dataset.py  # Script to split dataset into training and testing
├── README.md  # Project documentation
└── requirements.txt
```
### Dataset
The dataset consists of two folders:

data/1/: Contains 500 snoring audio files, each 1 second long.
data/0/: Contains 500 non-snoring audio files, each 1 second long. These include background sounds like baby crying, clock ticking, door opening, etc.
Dataset Sources
The snoring sounds were collected from the following online sources:

Soundsnap - Snoring
Zapsplat - Snoring
Fesliyan Studios - Snoring
YouTube - Snoring
YouTube - Snoring
Project Workflow

### 1. Data Preprocessing
The first step is to split the dataset into training and testing sets. The script split_dataset.py handles this task:
#### ==> python src/split_dataset.py
### 2. Feature Extraction
The MFCC (Mel-Frequency Cepstral Coefficients) features are extracted from the audio files. This is done using the preprocess.py and feature_extraction.py scripts.
#### ==> python src/preprocess.py
This script will:
Load the audio files.
Extract MFCC features.
Split the data into training and validation sets.
Save the processed data in .npz files for training and validation.

### 3. Model Training
The train.py script is used to train a neural network on the preprocessed data.
#### ==> python src/train.py
During training, the model weights that achieve the best validation loss are saved to the models/ directory. The training process is monitored using callbacks like ModelCheckpoint and EarlyStopping.

### 4. Model Evaluation
Once the model is trained, it can be evaluated on the test data using the evaluate.py script.
#### ==> python src/evaluate.py

### 5. Testing
After all process, the individual audios can be checked to infer whterher the audio is a snore or not.
#### ==> python src/inference.py
This script loads the saved model and evaluates its performance using metrics like accuracy, precision, recall, and F1-score.
____________________________________________________________________

A pipeline main.py can also be used to run all the programs at once. But, currently, its not functional so, try to run every program manually.
____________________________________________________________________

Configuration
All configuration settings (e.g., paths, audio processing parameters, training parameters) are stored in config.py. You can modify this file to change the sample rate, number of MFCCs, batch size, number of epochs, learning rate, and other settings.

1. How to Use
Clone the Repository: Clone the project repository to your local machine.
git clone <git clone https://github.com/jibran-mujtaba/Snore_Detection.git>
>

2. Install Dependencies: Install the necessary Python packages.
pip install -r requirements.txt

3. Organize Your Dataset: Place your audio files in the data/1 and data/0 directories.

4. Split the Dataset: Run the split_dataset.py script to organize the data into training and testing sets.
==> python src/split_dataset.py

5. Preprocess the Data: Extract MFCC features and prepare the data for training.
==> python src/preprocess.py

6. Train the Model: Train the model using the processed data.
==> python src/train.py

7. Evaluate the Model: Evaluate the trained model on the test set.
==> python src/evaluate.py

Results
The final model performance will be printed after running the evaluate.py script. This includes metrics like accuracy, precision, recall, and F1-score.

Future Work
Potential improvements for the project include:

Data Augmentation: Implementing data augmentation techniques to increase the diversity of training data.
Hyperparameter Tuning: Fine-tuning the model's hyperparameters for better performance.
Model Optimization: Exploring more advanced architectures or techniques such as transfer learning.
Contributing
If you wish to contribute to this project, feel free to fork the repository and submit a pull request.


By JIBRAN MUJTABA(ML Researcher) ---> DEVSTER LABS
===============================================================================================================




