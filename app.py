import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="OMR Sheet Evaluator",
    page_icon="📝",
    layout="wide"
)

class SimplifiedOMRProcessor:
    """Simplified OMR processor optimized for your specific sheet format"""
    
    def __init__(self):
        self.questions_per_subject = 20
        self.total_questions = 100
        self.subjects = ['PYTHON', 'DATA ANALYSIS', 'MySQL', 'POWER BI', 'Adv STATS']
        
    def preprocess_image(self, image):
        """Preprocess the uploaded image"""
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold for better bubble detection
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return image, gray, thresh
    
    def detect_bubbles(self, thresh_image):
        """Detect all bubbles in the image"""
        # Find contours
        contours, _ = cv2.findContours(
            thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 500:  # Filter by area
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (bubbles should be roughly circular)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:  # Filter non-circular shapes
                    continue
            
            bubbles.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + w // 2,
                'cy': y + h // 2,
                'area': area,
                'contour': contour
            })
        
        return bubbles
    
    def organize_bubbles_into_grid(self, bubbles):
        """Organize detected bubbles into a grid structure"""
        if not bubbles:
            return {}
        
        # Sort bubbles by position
        bubbles = sorted(bubbles, key=lambda b: (b['cy'], b['cx']))
        
        # Group bubbles into rows
        rows = []
        current_row = []
        row_y = bubbles[0]['cy'] if bubbles else 0
        row_threshold = 15  # pixels tolerance for same row
        
        for bubble in bubbles:
            if abs(bubble['cy'] - row_y) <= row_threshold:
                current_row.append(bubble)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b['cx']))
                current_row = [bubble]
                row_y = bubble['cy']
        
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['cx']))
        
        # Group into questions (4 bubbles per question, 5 questions per row)
        questions = {}
        question_num = 1
        
        for row in rows:
            # Process groups of 4 bubbles (A, B, C, D options)
            for i in range(0, len(row), 4):
                if i + 3 < len(row) and question_num <= 100:
                    questions[question_num] = row[i:i+4]
                    question_num += 1
        
        return questions
    
    def classify_bubble(self, image, bubble, threshold_value=0.3):
        """Determine if a bubble is filled based on pixel intensity"""
        x, y, w, h = bubble['x'], bubble['y'], bubble['w'], bubble['h']
        
        # Extract ROI with padding
        padding = 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Calculate the percentage of dark pixels
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels == 0:
            return False
            
        dark_pixels = np.sum(binary == 0)
        fill_ratio = dark_pixels / total_pixels
        
        # Bubble is considered filled if more than threshold of pixels are dark
        return fill_ratio > threshold_value
    
    def extract_answers(self, image, questions):
        """Extract answers from detected questions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        answers = {}
        
        for question_num, bubbles in questions.items():
            if len(bubbles) != 4:
                answers[question_num] = 'INVALID'
                continue
            
            # Check which bubble is filled
            filled_indices = []
            fill_scores = []
            
            for i, bubble in enumerate(bubbles):
                is_filled = self.classify_bubble(gray, bubble)
                if is_filled:
                    filled_indices.append(i)
                    
                    # Calculate fill score for disambiguation
                    x, y, w, h = bubble['x'], bubble['y'], bubble['w'], bubble['h']
                    roi = gray[y:y+h, x:x+w]
                    if roi.size > 0:
                        fill_score = np.mean(255 - roi)
                        fill_scores.append(fill_score)
            
            # Determine answer
            options = ['A', 'B', 'C', 'D']
            if len(filled_indices) == 1:
                answers[question_num] = options[filled_indices[0]]
            elif len(filled_indices) > 1:
                # Multiple bubbles filled - choose the darkest one
                if fill_scores:
                    best_idx = filled_indices[np.argmax(fill_scores)]
                    answers[question_num] = options[best_idx]
                else:
                    answers[question_num] = 'MULTIPLE'
            else:
                answers[question_num] = 'BLANK'
        
        return answers
    
    def process_image(self, image):
        """Main processing pipeline"""
        # Preprocess
        original, gray, thresh = self.preprocess_image(image)
        
        # Detect bubbles
        bubbles = self.detect_bubbles(thresh)
        
        # Organize into grid
        questions = self.organize_bubbles_into_grid(bubbles)
        
        # Extract answers
        answers = self.extract_answers(original, questions)
        
        # Fill missing questions
        for q in range(1, 101):
            if q not in answers:
                answers[q] = 'NOT_DETECTED'
        
        return {
            'answers': answers,
            'bubbles_detected': len(bubbles),
            'questions_detected': len(questions),
            'image_shape': original.shape
        }
    
    def calculate_score(self, student_answers, answer_key):
        """Calculate score based on answer key"""
        correct = 0
        wrong = 0
        blank = 0
        
        details = []
        
        for q_num in range(1, 101):
            student_ans = student_answers.get(q_num, 'BLANK')
            correct_ans = answer_key.get(q_num, '')
            
            if student_ans == 'BLANK' or student_ans == 'NOT_DETECTED':
                blank += 1
                status = 'Blank'
            elif student_ans == 'MULTIPLE':
                wrong += 1
                status = 'Multiple'
            elif student_ans == correct_ans:
                correct += 1
                status = '✓'
            else:
                wrong += 1
                status = '✗'
            
            details.append({
                'Question': q_num,
                'Student Answer': student_ans,
                'Correct Answer': correct_ans,
                'Status': status
            })
        
        return {
            'correct': correct,
            'wrong': wrong,
            'blank': blank,
            'total_score': correct,
            'percentage': (correct / 100) * 100,
            'details': details
        }

def create_sample_answer_key():
    """Create a sample answer key"""
    # This is a sample - replace with your actual answer key
    answer_key = {}
    options = ['A', 'B', 'C', 'D']
    
    # Generate sample answers (you should replace this with actual answer key)
    for i in range(1, 101):
        answer_key[i] = options[(i - 1) % 4]
    
    return answer_key

def main():
    st.title("📝 OMR Sheet Evaluator")
    st.markdown("### Automatic evaluation of 100-question OMR sheets")
    
    # Initialize processor
    processor = SimplifiedOMRProcessor()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Answer Key")
        answer_key_option = st.radio(
            "Select answer key source:",
            ["Use Sample Key", "Upload Answer Key", "Manual Entry"]
        )
        
        answer_key = {}
        
        if answer_key_option == "Use Sample Key":
            answer_key = create_sample_answer_key()
            st.success("Sample answer key loaded")
            
        elif answer_key_option == "Upload Answer Key":
            key_file = st.file_uploader("Upload answer key (JSON)", type=['json'])
            if key_file:
                answer_key = json.load(key_file)
                st.success(f"Loaded {len(answer_key)} answers")
                
        else:  # Manual Entry
            st.write("Enter answers for each question:")
            cols = st.columns(4)
            for i in range(1, 101):
                col_idx = (i - 1) % 4
                with cols[col_idx]:
                    answer_key[i] = st.selectbox(
                        f"Q{i}", 
                        ['A', 'B', 'C', 'D'],
                        key=f"q_{i}"
                    )
        
        st.subheader("Detection Settings")
        threshold = st.slider(
            "Fill Detection Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Lower values detect lighter marks"
        )
    
    # Main content area
    st.header("Upload OMR Sheet")
    
    uploaded_file = st.file_uploader(
        "Choose an OMR sheet image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of the filled OMR sheet"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process button
        if st.button("🔍 Process OMR Sheet", type="primary"):
            with st.spinner("Processing image..."):
                # Process the image
                results = processor.process_image(image)
                
                # Calculate score if answer key is available
                if answer_key:
                    score_results = processor.calculate_score(
                        results['answers'], 
                        answer_key
                    )
                    
                    with col2:
                        st.subheader("Results")
                        
                        # Display metrics
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("Correct", score_results['correct'])
                        with met_col2:
                            st.metric("Wrong", score_results['wrong'])
                        with met_col3:
                            st.metric("Blank", score_results['blank'])
                        
                        # Overall score
                        st.success(f"**Total Score: {score_results['total_score']}/100 ({score_results['percentage']:.1f}%)**")
                        
                        # Subject-wise breakdown
                        st.subheader("Subject-wise Performance")
                        subjects = processor.subjects
                        subject_scores = {}
                        
                        for i, subject in enumerate(subjects):
                            start_q = i * 20 + 1
                            end_q = start_q + 19
                            
                            correct_in_subject = sum(
                                1 for q in range(start_q, end_q + 1)
                                if results['answers'].get(q) == answer_key.get(q)
                            )
                            subject_scores[subject] = correct_in_subject
                        
                        subject_df = pd.DataFrame(
                            subject_scores.items(),
                            columns=['Subject', 'Score (out of 20)']
                        )
                        st.dataframe(subject_df)
                    
                    # Detection statistics
                    with st.expander("Detection Statistics"):
                        st.write(f"Total bubbles detected: {results['bubbles_detected']}")
                        st.write(f"Questions organized: {results['questions_detected']}")
                        st.write(f"Image dimensions: {results['image_shape']}")
                    
                    # Detailed answers
                    with st.expander("Detailed Answer Sheet"):
                        details_df = pd.DataFrame(score_results['details'])
                        
                        # Display in groups of 20
                        for i in range(0, 100, 20):
                            st.write(f"**Questions {i+1}-{i+20}**")
                            st.dataframe(
                                details_df.iloc[i:i+20],
                                use_container_width=True
                            )
                    
                    # Export results
                    st.subheader("Export Results")
                    
                    # Prepare export data
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'score': score_results['total_score'],
                        'percentage': score_results['percentage'],
                        'correct': score_results['correct'],
                        'wrong': score_results['wrong'],
                        'blank': score_results['blank'],
                        'answers': results['answers'],
                        'subject_scores': subject_scores
                    }
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                else:
                    st.warning("Please provide an answer key to calculate scores")
                    
                    # Show detected answers
                    with col2:
                        st.subheader("Detected Answers")
                        answers_df = pd.DataFrame(
                            results['answers'].items(),
                            columns=['Question', 'Answer']
                        )
                        st.dataframe(answers_df)

if __name__ == "__main__":
    main()
