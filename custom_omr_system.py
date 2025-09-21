import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import logging
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Optional
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OMRAnswerKeyLoader:
    """Load and manage answer keys from Excel file"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.answer_keys = {}
        self.load_answer_keys()
    
    def load_answer_keys(self):
        """Load answer keys from Excel file"""
        try:
            # Read Excel file
            df = pd.read_excel(self.excel_path)
            
            # Extract answer keys for each set
            sets = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']  # Column headers
            
            for i, set_name in enumerate(sets):
                if set_name in df.columns:
                    answers = []
                    for idx, row in df.iterrows():
                        if idx == 0:  # Skip header row
                            continue
                        answer_cell = str(row[set_name]).strip()
                        # Extract answer (last character after '-')
                        if '-' in answer_cell:
                            answer = answer_cell.split('-')[-1].strip().upper()
                            if answer in ['A', 'B', 'C', 'D']:
                                answers.append(answer)
                    
                    if len(answers) > 0:
                        self.answer_keys[set_name] = answers
                        logger.info(f"Loaded {len(answers)} answers for {set_name}")
            
            # Create unified answer key (assuming all subjects combined = 100 questions)
            if len(self.answer_keys) > 0:
                unified_answers = []
                for set_name in sets:
                    if set_name in self.answer_keys:
                        unified_answers.extend(self.answer_keys[set_name])
                
                # Pad to 100 questions if needed
                while len(unified_answers) < 100:
                    unified_answers.extend(unified_answers[:min(20, 100-len(unified_answers))])
                
                self.answer_keys['SET_A'] = unified_answers[:100]
                self.answer_keys['SET_B'] = unified_answers[:100]  # Assuming same pattern
                
                logger.info(f"Created unified answer keys with {len(unified_answers)} total questions")
        
        except Exception as e:
            logger.error(f"Error loading answer keys: {e}")
            # Fallback answer key
            self.answer_keys = {
                'SET_A': ['A', 'B', 'C', 'D'] * 25,
                'SET_B': ['B', 'C', 'D', 'A'] * 25
            }

class AdvancedImagePreprocessor:
    """Advanced image preprocessing for OMR sheets"""
    
    def __init__(self, target_size=(800, 1200)):
        self.target_size = target_size
    
    def detect_and_correct_skew(self, image):
        """Detect and correct skew in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:  # Use first 10 lines
                angle = theta * 180 / np.pi
                # Convert to degrees from vertical
                if angle > 90:
                    angle = angle - 180
                angles.append(angle)
            
            # Calculate median angle
            if angles:
                median_angle = np.median(angles)
                
                # Rotate image if skew is significant
                if abs(median_angle) > 0.5:
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, 
                                         (image.shape[1], image.shape[0]),
                                         borderValue=(255, 255, 255))
        
        return image
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def detect_sheet_boundary(self, image):
        """Detect OMR sheet boundary and crop"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the sheet)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop image
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image
    
    def preprocess_image(self, image_path: str):
        """Complete preprocessing pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect and correct skew
        image = self.detect_and_correct_skew(image)
        
        # Enhance contrast
        image = self.enhance_contrast(image)
        
        # Detect and crop sheet boundary
        image = self.detect_sheet_boundary(image)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        return image

class BubbleDetectorAdvanced:
    """Advanced bubble detection using template matching and contour analysis"""
    
    def __init__(self, bubble_size_range=(15, 35)):
        self.bubble_size_range = bubble_size_range
        self.bubble_template = self.create_bubble_template()
    
    def create_bubble_template(self):
        """Create a template bubble for matching"""
        size = 30
        template = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw circle outline
        cv2.circle(template, (size//2, size//2), size//3, 0, 2)
        
        return template
    
    def detect_grid_structure(self, image):
        """Detect the grid structure of the OMR sheet"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        grid_lines = cv2.add(horizontal_lines, vertical_lines)
        
        return grid_lines
    
    def detect_bubbles_by_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
        # Use adaptive threshold instead of OTSU for better results
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
        
            # Adjusted area range for your specific bubbles
            if 50 < area < 500:  # Changed from 100 < area < 1000
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                    # Lowered circularity threshold
                    if circularity > 0.5:  # Changed from 0.3
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                    
                        if 0.7 < aspect_ratio < 1.3:
                            bubble_roi = gray[y:y+h, x:x+w]
                            bubbles.append({
                                'contour': contour,
                                'bbox': (x, y, w, h),
                                'center': (x + w//2, y + h//2),
                                'area': area,
                                'circularity': circularity,
                                'roi': bubble_roi
                            })
    
    # Sort bubbles properly
    bubbles.sort(key=lambda b: (b['center'][1] // 20, b['center'][0]))
    return bubbles
    
    def organize_bubbles_into_grid(self, bubbles, questions_per_row=5):
        """Organize bubbles into question-answer grid"""
        if not bubbles:
            return {}
        
        # Group bubbles by rows (based on y-coordinate)
        rows = {}
        row_tolerance = 25  # pixels
        
        for bubble in bubbles:
            y_center = bubble['center'][1]
            
            # Find existing row or create new one
            assigned_row = None
            for row_y in rows.keys():
                if abs(y_center - row_y) <= row_tolerance:
                    assigned_row = row_y
                    break
            
            if assigned_row is None:
                assigned_row = y_center
                rows[assigned_row] = []
            
            rows[assigned_row].append(bubble)
        
        # Sort rows by y-coordinate
        sorted_rows = sorted(rows.items())
        
        # Organize into questions
        questions = {}
        question_num = 1
        
        for row_y, row_bubbles in sorted_rows:
            # Sort bubbles in row by x-coordinate
            row_bubbles.sort(key=lambda b: b['center'][0])
            
            # Group into questions (assuming 4 options per question)
            for i in range(0, len(row_bubbles), 4):
                if i + 3 < len(row_bubbles):  # Ensure we have 4 bubbles
                    questions[question_num] = row_bubbles[i:i+4]
                    question_num += 1
                    
                    if question_num > 100:  # Limit to 100 questions
                        break
        
        return questions

class BubbleCNNClassifier:
    """CNN classifier for determining if bubbles are filled"""
    
    def __init__(self, input_size=32):
        self.input_size = input_size
        self.model = None
        self.is_trained = False
    
    def create_model(self):
        """Create CNN architecture optimized for bubble classification"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(self.input_size, self.input_size, 1)),
            
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten
            keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            
            # Output layer - 3 classes: empty, filled, unclear
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_bubble_roi(self, roi):
        """Preprocess bubble ROI for classification"""
        # Resize to input size
        resized = cv2.resize(roi, (self.input_size, self.input_size))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add channel dimension
        return np.expand_dims(normalized, axis=-1)
    
    def classify_bubble(self, bubble_roi):
        """Classify a single bubble ROI"""
        if self.model is None:
            return self._traditional_classification(bubble_roi)
        
        processed_roi = self.preprocess_bubble_roi(bubble_roi)
        prediction = self.model.predict(np.expand_dims(processed_roi, 0), verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 0: empty, 1: filled, 2: unclear
        classes = ['empty', 'filled', 'unclear']
        
        return classes[class_idx], confidence
    
    def _traditional_classification(self, roi):
        """Traditional bubble classification as fallback"""
        # Apply threshold
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate filled percentage
        filled_pixels = np.sum(thresh > 0)
        total_pixels = roi.shape[0] * roi.shape[1]
        fill_ratio = filled_pixels / total_pixels
        
        if fill_ratio > 0.4:  # More than 40% filled
            return 'filled', fill_ratio
        elif fill_ratio > 0.15:  # Partially filled
            return 'unclear', fill_ratio
        else:
            return 'empty', 1 - fill_ratio
    
    def train_with_synthetic_data(self, num_samples=10000):
        """Train the model using synthetic bubble data"""
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            # Generate synthetic bubble
            bubble_type = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # empty, filled, unclear
            
            bubble_img = self._generate_synthetic_bubble(bubble_type)
            processed_bubble = self.preprocess_bubble_roi(bubble_img)
            
            X.append(processed_bubble)
            y.append(bubble_type)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        self.create_model()
        
        # Train
        logger.info("Training CNN classifier...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ],
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def _generate_synthetic_bubble(self, bubble_type):
        """Generate synthetic bubble image"""
        size = self.input_size
        img = np.ones((size, size), dtype=np.uint8) * 255
        
        center = size // 2
        radius = size // 3
        
        # Draw circle outline
        cv2.circle(img, (center, center), radius, 100, 2)
        
        if bubble_type == 1:  # Filled
            cv2.circle(img, (center, center), radius - 3, 50, -1)
        elif bubble_type == 2:  # Unclear/partial
            # Add some random marks
            for _ in range(np.random.randint(2, 5)):
                x = np.random.randint(center - radius//2, center + radius//2)
                y = np.random.randint(center - radius//2, center + radius//2)
                cv2.circle(img, (x, y), np.random.randint(2, 5), 150, -1)
        
        # Add noise
        noise = np.random.normal(0, 10, (size, size))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class OMRProcessor:
    """Main OMR processing pipeline"""
    
    def __init__(self, answer_key_path: str, samples_folder: str):
        self.samples_folder = samples_folder
        
        # Initialize components
        self.answer_key_loader = OMRAnswerKeyLoader(answer_key_path)
        self.preprocessor = AdvancedImagePreprocessor()
        self.bubble_detector = BubbleDetectorAdvanced()
        self.classifier = BubbleCNNClassifier()
        
        # Try to load existing model
        model_path = 'omr_bubble_classifier.h5'
        if not self.classifier.load_model(model_path):
            logger.info("No pre-trained model found. Training new model...")
            self.train_classifier(model_path)
    
    def train_classifier(self, model_path):
        """Train the bubble classifier"""
        history = self.classifier.train_with_synthetic_data(num_samples=20000)
        self.classifier.save_model(model_path)
        return history
    
    def process_sample_folder(self, set_name='Set A'):
        """Process all samples in a folder"""
        set_folder = os.path.join(self.samples_folder, set_name)
        
        if not os.path.exists(set_folder):
            logger.error(f"Folder not found: {set_folder}")
            return []
        
        results = []
        image_files = [f for f in os.listdir(set_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        logger.info(f"Processing {len(image_files)} images from {set_name}")
        
        for image_file in image_files:
            image_path = os.path.join(set_folder, image_file)
            
            try:
                result = self.process_single_sheet(image_path, set_name.replace(' ', '_'))
                result['image_file'] = image_file
                results.append(result)
                
                logger.info(f"Processed {image_file}: Score {result['total_score']}/100")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                results.append({
                    'image_file': image_file,
                    'error': str(e),
                    'total_score': 0
                })
        
        return results
    
    def process_single_sheet(self, image_path: str, sheet_version: str = 'SET_A'):
        """Process a single OMR sheet"""
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image_path)
        
        # Detect bubbles
        bubbles = self.bubble_detector.detect_bubbles_by_contours(processed_image)
        
        # Organize into questions
        questions = self.bubble_detector.organize_bubbles_into_grid(bubbles)
        
        # Extract answers
        extracted_answers = self._extract_answers_from_questions(questions)
        
        # Calculate scores
        scores = self._calculate_scores(extracted_answers, sheet_version)
        
        # Create result
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'sheet_version': sheet_version,
            'bubbles_detected': len(bubbles),
            'questions_detected': len(questions),
            'extracted_answers': extracted_answers,
            'scores': scores,
            'total_score': scores['total_score']
        }
        
        return result
    
    def _extract_answers_from_questions(self, questions):
        """Extract answers from organized questions"""
        answers = {}
        
        for question_num, bubble_list in questions.items():
            if len(bubble_list) != 4:
                answers[question_num] = 'INVALID'
                continue
            
            filled_options = []
            option_labels = ['A', 'B', 'C', 'D']
            
            for i, bubble in enumerate(bubble_list):
                classification, confidence = self.classifier.classify_bubble(bubble['roi'])
                
                if classification == 'filled' and confidence > 0.6:
                    filled_options.append(option_labels[i])
            
            # Determine final answer
            if len(filled_options) == 1:
                answers[question_num] = filled_options[0]
            elif len(filled_options) == 0:
                answers[question_num] = 'BLANK'
            else:
                answers[question_num] = 'MULTIPLE'
        
        return answers
    
    def _calculate_scores(self, extracted_answers, sheet_version):
        """Calculate scores based on answer key"""
        if sheet_version not in self.answer_key_loader.answer_keys:
            logger.warning(f"Answer key not found for {sheet_version}")
            return {'total_score': 0, 'subject_scores': {}}
        
        correct_answers = self.answer_key_loader.answer_keys[sheet_version]
        
        # Subject mapping (20 questions each)
        subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
        subject_scores = {subject: 0 for subject in subjects}
        
        total_correct = 0
        
        for question_num, student_answer in extracted_answers.items():
            if 1 <= question_num <= len(correct_answers):
                correct_answer = correct_answers[question_num - 1]
                
                if student_answer == correct_answer:
                    total_correct += 1
                    
                    # Determine subject (20 questions per subject)
                    subject_index = (question_num - 1) // 20
                    if subject_index < len(subjects):
                        subject = subjects[subject_index]
                        subject_scores[subject] += 1
        
        # Convert to scores out of 20 for each subject
        for subject in subject_scores:
            subject_scores[subject] = (subject_scores[subject] / 20) * 20  # Already correct
        
        return {
            'total_score': total_correct,
            'subject_scores': subject_scores,
            'accuracy': (total_correct / 100) * 100 if len(correct_answers) > 0 else 0
        }
    
    def generate_report(self, results):
        """Generate comprehensive report"""
        if not results:
            return "No results to report"
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        if not valid_results:
            return f"All {len(error_results)} files failed to process"
        
        # Calculate statistics
        scores = [r['total_score'] for r in valid_results]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # Subject-wise analysis
        subject_analysis = {}
        subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
        
        for subject in subjects:
            subject_scores = []
            for result in valid_results:
                if subject in result['scores']['subject_scores']:
                    subject_scores.append(result['scores']['subject_scores'][subject])
            
            if subject_scores:
                subject_analysis[subject] = {
                    'avg': np.mean(subject_scores),
                    'std': np.std(subject_scores),
                    'max': np.max(subject_scores),
                    'min': np.min(subject_scores)
                }
        
        # Generate report
        report = f"""
OMR PROCESSING REPORT
=====================
Processed Files: {len(valid_results)}
Failed Files: {len(error_results)}

OVERALL STATISTICS:
- Average Score: {avg_score:.2f}/100 ({(avg_score/100)*100:.1f}%)
- Standard Deviation: {std_score:.2f}
- Highest Score: {max_score}/100
- Lowest Score: {min_score}/100

SUBJECT-WISE ANALYSIS:
"""
        
        for subject, stats in subject_analysis.items():
            report += f"""
{subject}:
  Average: {stats['avg']:.2f}/20 ({(stats['avg']/20)*100:.1f}%)
  Std Dev: {stats['std']:.2f}
  Range: {stats['min']:.0f} - {stats['max']:.0f}
"""
        
        if error_results:
            report += f"\n\nFAILED FILES:\n"
            for error_result in error_results:
                report += f"- {error_result['image_file']}: {error_result['error']}\n"
        
        return report

def main():
    """Main execution function"""
    # Configuration
    samples_folder = "samples"  # Your samples folder
    answer_key_path = "Key (Set A and B).xlsx"  # Your Excel answer key
    
    # Initialize processor
    processor = OMRProcessor(answer_key_path, samples_folder)
    
    # Process Set A
    logger.info("Processing Set A...")
    set_a_results = processor.process_sample_folder('Set A')
    
    # Process Set B
    logger.info("Processing Set B...")
    set_b_results = processor.process_sample_folder('Set B')
    
    # Combine results
    all_results = set_a_results + set_b_results
    
    # Generate and save reports
    report = processor.generate_report(all_results)
    print(report)
    
    # Save detailed results
    with open('omr_processing_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save report
    with open('omr_processing_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Processing complete! Results saved to omr_processing_results.json")
    logger.info("Report saved to omr_processing_report.txt")

if __name__ == "__main__":
    main()

