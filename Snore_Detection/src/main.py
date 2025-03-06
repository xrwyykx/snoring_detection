import os
import logging
from src import preprocess, train, evaluate
from config import DATA_DIR, MODEL_DIR, LOG_DIR

def setup_logging():
    """Set up logging configuration."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(filename=os.path.join(LOG_DIR, 'process.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    
    logging.info("Starting the preprocessing step.")
    try:
        preprocess.preprocess_data()
        logging.info("Preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return
    
    logging.info("Starting the training step.")
    try:
        train.train_model()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return
    
    logging.info("Starting the evaluation step.")
    try:
        evaluate.evaluate_model()
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return

if __name__ == "__main__":
    main()
