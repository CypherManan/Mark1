import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import uuid
from PIL import Image
import logging
import base64
import time

import os
print("Current working dir:", os.getcwd())
print("Model exists:", os.path.exists('omr_bubble_classifier.h5'))

# Import your custom OMR classes
from custom_omr_system import (
    OMRProcessor, OMRAnswerKeyLoader, 
    AdvancedImagePreprocessor, BubbleDetectorAdvanced, 
    BubbleCNNClassifier
)
from omr_training_pipeline import EnhancedBubbleClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .upload-zone {
        border: 3px dashed rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        margin: 2rem 0;
    }
    
    .result-card {
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .subject-score {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    h1, h2, h3 {
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.enhanced_classifier = None
    st.session_state.processing_history = []
    st.session_state.total_processed = 0
    st.session_state.average_score = 0

@st.cache_resource
def initialize_omr_system():
    """Initialize OMR system components"""
    try:
        # Check if answer key exists
        if not os.path.exists('Key (Set A and B).xlsx'):
            st.error("Answer key file 'Key (Set A and B).xlsx' not found!")
            return None, None
        
        # Initialize enhanced classifier
        enhanced_classifier = EnhancedBubbleClassifier()
        
        # Try to load trained model
        model_paths = ['enhanced_omr_classifier.h5', 'best_omr_classifier.h5', 'omr_bubble_classifier.h5']
        model_loaded = False
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    if enhanced_classifier.load_model(model_path):
                        logger.info(f"Loaded trained model from {model_path}")
                        model_loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Failed to load model {model_path}: {e}")
        
        if not model_loaded:
            st.warning("No trained model found. Using traditional detection methods.")
        
        # Initialize OMR processor
        omr_processor = OMRProcessor('Key (Set A and B).xlsx', 'samples')
        if enhanced_classifier and model_loaded:
            omr_processor.classifier = enhanced_classifier
        
        return omr_processor, enhanced_classifier
    
    except Exception as e:
        st.error(f"Error initializing OMR system: {e}")
        return None, None

def process_omr_image(image, sheet_version, student_id=""):
    """Process a single OMR sheet"""
    try:
        # Save image temporarily
        temp_filename = f"temp_{uuid.uuid4()}.png"
        cv2.imwrite(temp_filename, image)
        
        # Process with OMR system
        if st.session_state.processor:
            start_time = datetime.now()
            result = st.session_state.processor.process_single_sheet(temp_filename, sheet_version)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add metadata
            result['student_id'] = student_id
            result['processing_time'] = processing_time
            result['percentage'] = (result.get('total_score', 0) / 100) * 100
            
            # Clean up temp file
            os.remove(temp_filename)
            
            return result
        else:
            os.remove(temp_filename)
            return None
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None

def display_results(result):
    """Display processing results"""
    if not result:
        st.error("Processing failed. Please check the image and try again.")
        return
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Score", f"{result.get('total_score', 0)}/100")
    with col2:
        st.metric("Percentage", f"{result.get('percentage', 0):.1f}%")
    with col3:
        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
    
    st.markdown("### 📊 Subject-wise Scores")
    
    if 'scores' in result and 'subject_scores' in result['scores']:
        subjects = result['scores']['subject_scores']
        
        # Create a DataFrame for better visualization
        df_subjects = pd.DataFrame([
            {'Subject': subject, 'Score': score, 'Out of': 20, 'Percentage': (score/20)*100}
            for subject, score in subjects.items()
        ])
        
        st.dataframe(df_subjects, use_container_width=True, hide_index=True)
        
        # Bar chart
        st.bar_chart(df_subjects.set_index('Subject')['Score'])
    
    with st.expander("📋 Detailed Information"):
        st.write(f"**Bubbles Detected:** {result.get('bubbles_detected', 0)}")
        st.write(f"**Questions Detected:** {result.get('questions_detected', 0)}")
        st.write(f"**Sheet Version:** {result.get('sheet_version', 'Unknown')}")
        if result.get('student_id'):
            st.write(f"**Student ID:** {result.get('student_id')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;">
            🎓 OMR Evaluation System
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.9; color: white;">
            Advanced optical mark recognition with AI-powered bubble detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state.processor is None:
        with st.spinner("Initializing OMR system..."):
            processor, classifier = initialize_omr_system()
            st.session_state.processor = processor
            st.session_state.enhanced_classifier = classifier
    
    # Statistics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">System Status</div>
            <div class="stat-value">✅</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_status = "🤖" if (st.session_state.enhanced_classifier and 
                               st.session_state.enhanced_classifier.is_trained) else "📊"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">AI Model</div>
            <div class="stat-value">{model_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Processed</div>
            <div class="stat-value">{st.session_state.total_processed}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Avg Score</div>
            <div class="stat-value">{st.session_state.average_score:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📁 Batch Processing", "📊 Analytics"])
    
    with tab1:
        st.markdown("### Upload OMR Sheet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an OMR sheet image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the OMR sheet"
            )
            
            if uploaded_file is not None:
                # Convert to OpenCV format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Display image
                st.image(image, channels="BGR", caption="Uploaded OMR Sheet", use_column_width=True)
        
        with col2:
            st.markdown("### Processing Options")
            
            sheet_version = st.selectbox(
                "Sheet Version",
                ["SET_A", "SET_B"],
                help="Select the version of the OMR sheet"
            )
            
            student_id = st.text_input(
                "Student ID (Optional)",
                placeholder="Enter student ID"
            )
            
            if st.button("🎯 Process OMR Sheet", type="primary"):
                if uploaded_file is not None:
                    with st.spinner("Processing... This may take a moment"):
                        result = process_omr_image(image, sheet_version, student_id)
                        
                        if result:
                            st.success("✅ Processing completed successfully!")
                            
                            # Update statistics
                            st.session_state.total_processed += 1
                            st.session_state.processing_history.append(result)
                            
                            # Calculate new average
                            total_scores = sum(r.get('total_score', 0) for r in st.session_state.processing_history)
                            st.session_state.average_score = (total_scores / len(st.session_state.processing_history))
                            
                            # Display results
                            display_results(result)
                            
                            # Download results as JSON
                            result_json = json.dumps(result, indent=2)
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=result_json,
                                file_name=f"omr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        else:
                            st.error("Processing failed. Please check the image and try again.")
                else:
                    st.warning("Please upload an image first.")
    
    with tab2:
        st.markdown("### Batch Processing")
        st.info("Upload multiple OMR sheets for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple OMR sheet images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            
            batch_sheet_version = st.selectbox(
                "Sheet Version for Batch",
                ["SET_A", "SET_B"],
                key="batch_version"
            )
            
            if st.button("🚀 Process Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Convert to OpenCV format
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Process image
                    result = process_omr_image(image, batch_sheet_version)
                    if result:
                        result['filename'] = uploaded_file.name
                        batch_results.append(result)
                
                progress_bar.progress(1.0)
                status_text.text("Batch processing completed!")
                
                if batch_results:
                    # Display summary
                    st.success(f"✅ Processed {len(batch_results)} sheets successfully!")
                    
                    # Create summary DataFrame
                    summary_df = pd.DataFrame([
                        {
                            'File': r.get('filename', 'Unknown'),
                            'Score': r.get('total_score', 0),
                            'Percentage': f"{r.get('percentage', 0):.1f}%"
                        }
                        for r in batch_results
                    ])
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Download batch results
                    batch_json = json.dumps(batch_results, indent=2)
                    st.download_button(
                        label="📥 Download Batch Results (JSON)",
                        data=batch_json,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with tab3:
        st.markdown("### Analytics Dashboard")
        
        if st.session_state.processing_history:
            # Create analytics
            df_history = pd.DataFrame(st.session_state.processing_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Score Distribution")
                scores = [r.get('total_score', 0) for r in st.session_state.processing_history]
                st.bar_chart(pd.Series(scores).value_counts().sort_index())
            
            with col2:
                st.markdown("#### Processing Times")
                times = [r.get('processing_time', 0) for r in st.session_state.processing_history]
                st.line_chart(times)
            
            # Subject-wise analysis
            st.markdown("#### Subject Performance Overview")
            
            all_subjects = {}
            for result in st.session_state.processing_history:
                if 'scores' in result and 'subject_scores' in result['scores']:
                    for subject, score in result['scores']['subject_scores'].items():
                        if subject not in all_subjects:
                            all_subjects[subject] = []
                        all_subjects[subject].append(score)
            
            if all_subjects:
                subject_stats = []
                for subject, scores in all_subjects.items():
                    subject_stats.append({
                        'Subject': subject,
                        'Average': np.mean(scores),
                        'Min': min(scores),
                        'Max': max(scores),
                        'Std Dev': np.std(scores)
                    })
                
                df_subjects = pd.DataFrame(subject_stats)
                st.dataframe(df_subjects, use_container_width=True, hide_index=True)
        else:
            st.info("No processing history available yet. Process some OMR sheets to see analytics.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ System Information")
        
        if st.session_state.processor:
            st.success("✅ OMR Processor: Active")
        else:
            st.error("❌ OMR Processor: Not initialized")
        
        if st.session_state.enhanced_classifier and st.session_state.enhanced_classifier.is_trained:
            st.success("✅ AI Model: Trained")
        else:
            st.warning("⚠️ AI Model: Using basic detection")
        
        st.markdown("---")
        
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. **Upload** an OMR sheet image
        2. **Select** the sheet version (Set A or B)
        3. **Enter** student ID (optional)
        4. **Click** Process to evaluate
        5. **Download** results as needed
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Configuration")
        
        if st.button("Clear Processing History"):
            st.session_state.processing_history = []
            st.session_state.total_processed = 0
            st.session_state.average_score = 0
            st.success("History cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()

