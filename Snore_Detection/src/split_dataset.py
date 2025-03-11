import os
import shutil
from sklearn.model_selection import train_test_split


snoring_dir = 'D:\\hansheng\\snoring_detection\\Snore_Detection\\data\\1'
non_snoring_dir = 'D:\\hansheng\\snoring_detection\\Snore_Detection\\data\\0'

train_dir = 'D:\\hansheng\\snoring_detection\\Snore_Detection\\src\data\\train'
test_dir = 'D:\\hansheng\\snoring_detection\\Snore_Detection\\src\\data\\test'

os.makedirs(os.path.join(train_dir, 'snoring'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'non-snoring'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'snoring'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'non-snoring'), exist_ok=True)

def split_and_copy_files(source_dir, target_train_dir, target_test_dir, test_size=0.2):

    files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), target_train_dir)
        
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), target_test_dir)

split_and_copy_files(snoring_dir, os.path.join(train_dir, 'snoring'), os.path.join(test_dir, 'snoring'))
split_and_copy_files(non_snoring_dir, os.path.join(train_dir, 'non-snoring'), os.path.join(test_dir, 'non-snoring'))

print("Dataset split into training and testing sets successfully!")
