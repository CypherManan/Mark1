import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime
import argparse
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Tuple, Dict

# Import your custom OMR classes
from custom_omr_system import (
    OMRAnswerKeyLoader, AdvancedImagePreprocessor, 
    BubbleDetectorAdvanced, BubbleCNNClassifier
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataBubbleExtractor:
    """Extract bubble regions from your real OMR sheets for training"""
    
    def __init__(self, samples_folder: str, answer_key_path: str):
        self.samples_folder = samples_folder
        self.answer_key_loader = OMRAnswerKeyLoader(answer_key_path)
        self.preprocessor = AdvancedImagePreprocessor()
        self.bubble_detector = BubbleDetectorAdvanced()
    
    def extract_labeled_bubbles_from_sheets(self, set_name='Set A', max_sheets=5):
        """Extract bubbles with labels from real OMR sheets"""
        set_folder = os.path.join(self.samples_folder, set_name)
        
        if not os.path.exists(set_folder):
            logger.error(f"Folder not found: {set_folder}")
            return []
        
        image_files = [f for f in os.listdir(set_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = image_files[:max_sheets]  # Limit number of sheets
        
        labeled_bubbles = []
        sheet_version = set_name.replace(' ', '_').upper()
        
        if sheet_version not in self.answer_key_loader.answer_keys:
            logger.error(f"No answer key found for {sheet_version}")
            return []
        
        correct_answers = self.answer_key_loader.answer_keys[sheet_version]
        
        logger.info(f"Extracting bubbles from {len(image_files)} sheets in {set_name}")
        
        for image_file in image_files:
            image_path = os.path.join(set_folder, image_file)
            
            try:
                # Preprocess image
                processed_image = self.preprocessor.preprocess_image(image_path)
                
                # Detect bubbles
                bubbles = self.bubble_detector.detect_bubbles_by_contours(processed_image)
                
                # Organize into questions
                questions = self.bubble_detector.organize_bubbles_into_grid(bubbles)
                
                # Extract labeled bubbles
                sheet_bubbles = self._extract_labeled_bubbles_from_questions(
                    questions, correct_answers, image_file
                )
                
                labeled_bubbles.extend(sheet_bubbles)
                
                logger.info(f"Extracted {len(sheet_bubbles)} labeled bubbles from {image_file}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        logger.info(f"Total labeled bubbles extracted: {len(labeled_bubbles)}")
        return labeled_bubbles
    
    def _extract_labeled_bubbles_from_questions(self, questions, correct_answers, image_file):
        """Extract bubbles with ground truth labels"""
        labeled_bubbles = []
        option_labels = ['A', 'B', 'C', 'D']
        
        for question_num, bubble_list in questions.items():
            if question_num > len(correct_answers) or len(bubble_list) != 4:
                continue
            
            correct_answer = correct_answers[question_num - 1]
            correct_option_idx = option_labels.index(correct_answer) if correct_answer in option_labels else -1
            
            for i, bubble in enumerate(bubble_list):
                # Label: 1 if this bubble should be filled, 0 if empty
                label = 1 if i == correct_option_idx else 0
                
                labeled_bubbles.append({
                    'roi': bubble['roi'],
                    'label': label,
                    'question_num': question_num,
                    'option': option_labels[i],
                    'image_file': image_file,
                    'bbox': bubble['bbox']
                })
        
        return labeled_bubbles
    
    def save_labeled_dataset(self, labeled_bubbles, output_folder='training_data'):
        """Save labeled dataset for training"""
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'filled'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'empty'), exist_ok=True)
        
        dataset_info = []
        
        for i, bubble_data in enumerate(labeled_bubbles):
            roi = bubble_data['roi']
            label = bubble_data['label']
            
            # Save image
            folder_name = 'filled' if label == 1 else 'empty'
            filename = f"bubble_{i:06d}.png"
            filepath = os.path.join(output_folder, folder_name, filename)
            
            cv2.imwrite(filepath, roi)
            
            # Save metadata
            dataset_info.append({
                'filename': filename,
                'label': label,
                'question_num': bubble_data['question_num'],
                'option': bubble_data['option'],
                'image_file': bubble_data['image_file'],
                'folder': folder_name
            })
        
        # Save dataset info
        with open(os.path.join(output_folder, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Saved {len(labeled_bubbles)} labeled samples to {output_folder}")
        return output_folder

class EnhancedBubbleClassifier(BubbleCNNClassifier):
    """Enhanced bubble classifier with real data training capabilities"""
    
    def __init__(self, input_size=32):
        super().__init__(input_size)
        self.training_history = None
    
    def create_enhanced_model(self):
        """Create an enhanced CNN model for better accuracy"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(self.input_size, self.input_size, 1)),
            
            # First block - Feature extraction
            keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second block - Pattern recognition
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third block - Complex features
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Fourth block - High-level features
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output layer - binary classification (filled/empty)
            keras.layers.Dense(2, activation='softmax')
        ])
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def load_real_training_data(self, training_data_folder):
        """Load real training data from extracted bubbles"""
        filled_folder = os.path.join(training_data_folder, 'filled')
        empty_folder = os.path.join(training_data_folder, 'empty')
        
        X = []
        y = []
        
        # Load filled bubbles
        if os.path.exists(filled_folder):
            for filename in os.listdir(filled_folder):
                if filename.lower().endswith('.png'):
                    img_path = os.path.join(filled_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed_img = self.preprocess_bubble_roi(img)
                        X.append(processed_img)
                        y.append(1)  # Filled
        
        # Load empty bubbles
        if os.path.exists(empty_folder):
            for filename in os.listdir(empty_folder):
                if filename.lower().endswith('.png'):
                    img_path = os.path.join(empty_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed_img = self.preprocess_bubble_roi(img)
                        X.append(processed_img)
                        y.append(0)  # Empty
        
        if len(X) == 0:
            raise ValueError("No training data found")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Loaded real training data: {len(X)} samples")
        logger.info(f"  Filled bubbles: {np.sum(y == 1)}")
        logger.info(f"  Empty bubbles: {np.sum(y == 0)}")
        
        return X, y
    
    def generate_augmented_data(self, X, y, augmentation_factor=3):
        """Generate augmented training data"""
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='constant',
            cval=1.0
        )
        
        augmented_X = []
        augmented_y = []
        
        # Keep original data
        augmented_X.extend(X)
        augmented_y.extend(y)
        
        # Generate augmented data
        for i in range(len(X)):
            img = X[i:i+1]  # Single image
            label = y[i]
            
            # Generate augmented versions
            aug_gen = datagen.flow(img, batch_size=1)
            
            for _ in range(augmentation_factor):
                aug_img = next(aug_gen)[0]
                augmented_X.append(aug_img)
                augmented_y.append(label)
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def train_on_real_data(self, training_data_folder, epochs=100, use_augmentation=True):
        """Train the model on real OMR bubble data"""
        # Load real data
        X_real, y_real = self.load_real_training_data(training_data_folder)
        
        # Generate synthetic data to supplement real data
        X_synthetic, y_synthetic = self._generate_synthetic_training_data(len(X_real) * 2)
        
        # Combine real and synthetic data
        X_combined = np.vstack([X_real, X_synthetic])
        y_combined = np.hstack([y_real, y_synthetic])
        
        # Apply data augmentation if requested
        if use_augmentation:
            logger.info("Applying data augmentation...")
            X_combined, y_combined = self.generate_augmented_data(X_combined, y_combined)
        
        logger.info(f"Final training dataset size: {len(X_combined)} samples")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Create enhanced model
        self.create_enhanced_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_omr_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Evaluate on test set
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Generate predictions for detailed analysis
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Print results
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Empty', 'Filled']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'training_time': training_time,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'confusion_matrix': cm,
            'history': self.training_history
        }
    
    def _generate_synthetic_training_data(self, num_samples):
        """Generate synthetic training data to supplement real data"""
        X = []
        y = []
        
        for i in range(num_samples):
            # Generate bubble type (50-50 split)
            is_filled = i % 2 == 0
            
            # Generate synthetic bubble
            bubble_img = self._generate_realistic_bubble(is_filled)
            processed_bubble = self.preprocess_bubble_roi(bubble_img)
            
            X.append(processed_bubble)
            y.append(1 if is_filled else 0)
        
        return np.array(X), np.array(y)
    
    def _generate_realistic_bubble(self, is_filled):
        """Generate realistic synthetic bubble"""
        size = self.input_size
        img = np.ones((size, size), dtype=np.uint8) * 255
        
        center = size // 2
        radius = random.randint(size // 4, size // 3)
        
        # Draw circle outline with varying thickness
        thickness = random.randint(1, 3)
        outline_color = random.randint(50, 150)
        cv2.circle(img, (center, center), radius, outline_color, thickness)
        
        if is_filled:
            # Fill the bubble with varying intensity
            fill_intensity = random.randint(30, 100)
            cv2.circle(img, (center, center), radius - thickness, fill_intensity, -1)
            
            # Add some texture
            for _ in range(random.randint(0, 3)):
                x = random.randint(center - radius//3, center + radius//3)
                y = random.randint(center - radius//3, center + radius//3)
                cv2.circle(img, (x, y), random.randint(1, 3), 
                          fill_intensity + random.randint(-20, 20), -1)
        
        # Add realistic noise and distortions
        # Gaussian noise
        noise = np.random.normal(0, random.uniform(5, 15), (size, size))
        img = np.clip(img + noise, 0, 255)
        
        # Random blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Slight rotation
        if random.random() < 0.4:
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            img = cv2.warpAffine(img, M, (size, size), borderValue=255)
        
        return img.astype(np.uint8)
    
    def plot_training_results(self):
        """Plot comprehensive training results"""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.training_history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.training_history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.training_history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.training_history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.training_history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.training_history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.training_history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

class OMRTrainingPipeline:
    """Complete training pipeline for your OMR system"""
    
    def __init__(self, samples_folder: str, answer_key_path: str):
        self.samples_folder = samples_folder
        self.answer_key_path = answer_key_path
        self.extractor = RealDataBubbleExtractor(samples_folder, answer_key_path)
        self.classifier = EnhancedBubbleClassifier()
    
    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting OMR Training Pipeline...")
        
        # Step 1: Extract labeled bubbles from real sheets
        logger.info("Step 1: Extracting labeled bubbles from real OMR sheets...")
        set_a_bubbles = self.extractor.extract_labeled_bubbles_from_sheets('Set A', max_sheets=10)
        set_b_bubbles = self.extractor.extract_labeled_bubbles_from_sheets('Set B', max_sheets=10)
        
        all_bubbles = set_a_bubbles + set_b_bubbles
        
        if len(all_bubbles) < 100:
            logger.warning(f"Only {len(all_bubbles)} bubbles extracted. Consider using more sheets for better training.")
        
        # Step 2: Save labeled dataset
        logger.info("Step 2: Saving labeled dataset...")
        training_data_folder = self.extractor.save_labeled_dataset(all_bubbles)
        
        # Step 3: Train enhanced classifier
        logger.info("Step 3: Training enhanced bubble classifier...")
        training_results = self.classifier.train_on_real_data(
            training_data_folder,
            epochs=150,
            use_augmentation=True
        )
        
        # Step 4: Save trained model
        model_path = 'enhanced_omr_classifier.h5'
        self.classifier.save_model(model_path)
        
        # Step 5: Generate training report
        report = self._generate_training_report(training_results, len(all_bubbles))
        
        # Save report
        with open('training_report.txt', 'w') as f:
            f.write(report)
        
        # Plot results
        self.classifier.plot_training_results()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training report saved to: training_report.txt")
        
        return training_results
    
    def _generate_training_report(self, training_results, num_real_bubbles):
        """Generate comprehensive training report"""
        report = f"""
OMR BUBBLE CLASSIFIER TRAINING REPORT
=====================================
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Time: {training_results['training_time']}

DATA STATISTICS:
- Real bubble samples extracted: {num_real_bubbles}
- Training accuracy: {max(training_results['history'].history['accuracy']):.4f}
- Validation accuracy: {max(training_results['history'].history['val_accuracy']):.4f}
- Test accuracy: {training_results['test_accuracy']:.4f}
- Test precision: {training_results['test_precision']:.4f}
- Test recall: {training_results['test_recall']:.4f}

PERFORMANCE METRICS:
- Final training accuracy: {training_results['history'].history['accuracy'][-1]:.4f}
- Final validation accuracy: {training_results['history'].history['val_accuracy'][-1]:.4f}
- Best validation accuracy: {max(training_results['history'].history['val_accuracy']):.4f}

CONFUSION MATRIX:
{training_results['confusion_matrix']}

MODEL ARCHITECTURE:
- Enhanced CNN with 4 convolutional blocks
- Data augmentation applied
- Real + synthetic data training
- Early stopping and learning rate reduction
- Model checkpoint saving

RECOMMENDATIONS:
- Model is ready for production use
- Expected accuracy on real OMR sheets: {training_results['test_accuracy']*100:.1f}%
- Consider retraining if accuracy drops below 95%
"""
        
        return report

def main():
    """Main function to run the training pipeline"""
    parser = argparse.ArgumentParser(description='Train OMR Bubble Classifier')
    parser.add_argument('--samples-folder', type=str, default='samples', 
                       help='Path to samples folder containing Set A and Set B')
    parser.add_argument('--answer-key', type=str, default='Key (Set A and B).xlsx',
                       help='Path to Excel answer key file')
    parser.add_argument('--max-sheets-per-set', type=int, default=10,
                       help='Maximum number of sheets to process per set')
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not os.path.exists(args.samples_folder):
        logger.error(f"Samples folder not found: {args.samples_folder}")
        return
    
    if not os.path.exists(args.answer_key):
        logger.error(f"Answer key file not found: {args.answer_key}")
        return
    
    # Initialize and run training pipeline
    pipeline = OMRTrainingPipeline(args.samples_folder, args.answer_key)
    
    try:
        results = pipeline.run_complete_training_pipeline()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Model saved as: enhanced_omr_classifier.h5")
        print("Check training_report.txt for detailed results")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
