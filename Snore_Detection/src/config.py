import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR,'..', 'models')
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')

# Dataset paths
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

# Training parameters(Adjust them accordingly)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


SR = 16000  
N_MFCC = 13 


N_MELS = 128 
HOP_LENGTH = 512 


TIME_STEPS = 1280   # adjust based on your model requirements


AUGMENTATION = True  # Enable/disable data augmentation
TIME_STRETCH_RATES = [0.8, 1.2] 
PITCH_SHIFT_STEPS = [-2, 2] 
NOISE_FACTOR = 0.005  


NORMALIZE = True  # Enable/disable normalization
