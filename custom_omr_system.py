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
            sets = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
            
            for i, set_name in enumerate(sets):
                if set_name in df.columns:
                    answers = []
                    for idx, row in df.iterrows():
                        if idx == 0:
                            continue
                        answer_cell = str(row[set_name]).strip()
                        if '-' in answer_cell:
                            answer = answer_cell.split('-')[-1].strip().upper()
                            if answer in ['A', 'B', 'C', 'D']:
                                answers.append(answer)
                    
                    if len(answers) > 0:
                        self.answer_keys[set_name] = answers
                        logger.info(f"Loaded {len(answers)} answers for {set_name}")
            
            if len(self.answer_keys) > 0:
                unified_answers = []
                for set_name in sets:
                    if set_name in self.answer_keys:
                        unified_answers.extend(self.answer_keys[set_name])
                
                while len(unified_answers) < 100:
                    unified_answers.extend(unified_answers[:min(20, 100-len(unified_answers))])
                
                self.answer_keys['SET_A'] = unified_answers[:100]
                self.answer_keys['SET_B'] = unified_answers[:100]
                
                logger.info(f"Created unified answer keys with {len(unified_answers)} total questions")
        
        except Exception as e:
            logger.error(f"Error loading answer keys: {e}")
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
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:
                angle = theta * 180 / np.pi
                if angle > 90:
                    angle = angle - 180
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def detect_sheet_boundary(self, image):
        """Detect OMR sheet boundary and crop"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image
    
    def preprocess_image(self, image_path: str):
        """Complete preprocessing pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = self.detect_and_correct_skew(image)
        image = self.enhance_contrast(image)
        image = self.detect_sheet_boundary(image)
        image = cv2.resize(image, self.target_size)
        
        return image

class BubbleDetectorAdvanced:
    """Advanced bubble detection using contour analysis"""
    
    def __init__(self, bubble_size_range=(15, 35)):
        self.bubble_size_range = bubble_size_range
    
    def detect_bubbles_by_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if (100 < area < 1000 and circularity > 0.3):
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
        
        bubbles.sort(key=lambda b: (b['center'][1] // 30, b['center'][0]))
        return bubbles
    
    def organize_bubbles_into_grid(self, bubbles, questions_per_row=5):
        if not bubbles:
            return {}
        
        rows = {}
        row_tolerance = 25
        
        for bubble in bubbles:
            y_center = bubble['center'][1]
            assigned_row = None
            for row_y in rows.keys():
                if abs(y_center - row_y) <= row_tolerance:
                    assigned_row = row_y
                    break
            
            if assigned_row is None:
                assigned_row = y_center
                rows[assigned_row] = []
            
            rows[assigned_row].append(bubble)
        
        sorted_rows = sorted(rows.items())
        
        questions = {}
        question_num = 1
        
        for row_y, row_bubbles in sorted_rows:
            row_bubbles.sort(key=lambda b: b['center'][0])
            for i in range(0, len(row_bubbles), 4):
                if i + 3 < len(row_bubbles):
                    questions[question_num] = row_bubbles[i:i+4]
                    question_num += 1
                    if question_num > 100:
                        break
        
        return questions

class BubbleCNNClassifier:
    """CNN classifier for determining if bubbles are filled"""
    
    def __init__(self, input_size=28): # Changed to 28x28 for consistency
        self.input_size = input_size
        self.model = None
        self.is_trained = False
    
    def create_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size, self.input_size, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid') # Corrected for binary
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy', # Corrected loss
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_bubble_roi(self, roi):
        resized = cv2.resize(roi, (self.input_size, self.input_size))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)
    
    def classify_bubble(self, bubble_roi):
        if self.model is None:
            return self._traditional_classification(bubble_roi)
        
        processed_roi = self.preprocess_bubble_roi(bubble_roi)
        prediction = self.model.predict(np.expand_dims(processed_roi, 0), verbose=0)
        
        # Binary classification based on a 0.5 threshold
        is_filled = prediction[0][0] > 0.5
        confidence = prediction[0][0] if is_filled else 1 - prediction[0][0]
        
        return 'filled' if is_filled else 'empty', confidence

    def _traditional_classification(self, roi):
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        filled_pixels = np.sum(thresh > 0)
        total_pixels = roi.shape[0] * roi.shape[1]
        fill_ratio = filled_pixels / total_pixels
        
        if fill_ratio > 0.4:
            return 'filled', fill_ratio
        else:
            return 'empty', 1 - fill_ratio
    
    def train_with_synthetic_data(self, num_samples=10000):
        # ... (This method would be in omr_training_pipeline.py, not here)
        raise NotImplementedError("This method is in the training pipeline file.")

    def load_model(self, filepath):
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
        self.answer_key_loader = OMRAnswerKeyLoader(answer_key_path)
        self.preprocessor = AdvancedImagePreprocessor()
        self.bubble_detector = BubbleDetectorAdvanced()
        self.classifier = BubbleCNNClassifier()
        
        model_path = 'omr_bubble_classifier.h5'
        if not self.classifier.load_model(model_path):
            logger.warning("No pre-trained model found. Using traditional classification.")
    
    def process_single_sheet(self, image_path: str, sheet_version: str = 'SET_A'):
        processed_image = self.preprocessor.preprocess_image(image_path)
        bubbles = self.bubble_detector.detect_bubbles_by_contours(processed_image)
        questions = self.bubble_detector.organize_bubbles_into_grid(bubbles)
        extracted_answers = self._extract_answers_from_questions(questions)
        scores = self._calculate_scores(extracted_answers, sheet_version)
        
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
        answers = {}
        option_labels = ['A', 'B', 'C', 'D']
        
        for question_num, bubble_list in questions.items():
            if len(bubble_list) != 4:
                answers[question_num] = 'INVALID'
                continue
            
            filled_options = []
            for i, bubble in enumerate(bubble_list):
                classification, confidence = self.classifier.classify_bubble(bubble['roi'])
                if classification == 'filled' and confidence > 0.5:
                    filled_options.append(option_labels[i])
            
            if len(filled_options) == 1:
                answers[question_num] = filled_options[0]
            elif len(filled_options) == 0:
                answers[question_num] = 'BLANK'
            else:
                answers[question_num] = 'MULTIPLE'
        
        return answers
    
    def _calculate_scores(self, extracted_answers, sheet_version):
        if sheet_version not in self.answer_key_loader.answer_keys:
            logger.warning(f"Answer key not found for {sheet_version}")
            return {'total_score': 0, 'subject_scores': {}}
        
        correct_answers = self.answer_key_loader.answer_keys[sheet_version]
        
        subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
        subject_scores = {subject: 0 for subject in subjects}
        total_correct = 0
        
        for question_num, student_answer in extracted_answers.items():
            if 1 <= question_num <= len(correct_answers):
                correct_answer = correct_answers[question_num - 1]
                if student_answer == correct_answer:
                    total_correct += 1
                    subject_index = (question_num - 1) // 20
                    if subject_index < len(subjects):
                        subject = subjects[subject_index]
                        subject_scores[subject] += 1
        
        return {
            'total_score': total_correct,
            'subject_scores': subject_scores
        }
    
