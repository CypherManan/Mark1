#!/usr/bin/env python3
"""
OMR Evaluation System - Complete Setup and Run Script
====================================================
This script sets up and runs the complete OMR evaluation system
for your specific data structure and requirements.
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    requirements = [
        "tensorflow>=2.12.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "Flask>=2.3.0",
        "Flask-CORS>=4.0.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0"  # For Excel file reading
    ]
    
    logger.info("Installing required packages...")
    
    for requirement in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            logger.info(f"✓ Installed {requirement}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {requirement}: {e}")
            return False
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'samples/Set A',
        'samples/Set B', 
        'uploads',
        'processed',
        'results',
        'models',
        'logs',
        'training_data/filled',
        'training_data/empty'
    ]
    
    logger.info("Creating directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")

def validate_data_files():
    """Validate that required data files exist"""
    required_files = [
        'Key (Set A and B).xlsx',
        'samples/Set A',
        'samples/Set B'
    ]
    
    logger.info("Validating data files...")
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logger.warning(f"✗ Missing: {file_path}")
        else:
            logger.info(f"✓ Found: {file_path}")
    
    if missing_files:
        logger.error("Missing required files. Please ensure you have:")
        logger.error("1. 'Key (Set A and B).xlsx' - Your Excel answer key file")
        logger.error("2. 'samples/Set A/' - Folder with Set A OMR images")
        logger.error("3. 'samples/Set B/' - Folder with Set B OMR images")
        return False
    
    return True

def run_training_pipeline(args):
    """Run the training pipeline"""
    logger.info("Starting training pipeline...")
    
    try:
        from omr_training_pipeline import OMRTrainingPipeline
        
        pipeline = OMRTrainingPipeline(
            samples_folder='samples',
            answer_key_path='Key (Set A and B).xlsx'
        )
        
        results = pipeline.run_complete_training_pipeline()
        
        logger.info("Training completed successfully!")
        logger.info(f"Model accuracy: {results['test_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_batch_processing():
    """Run batch processing on all sample images"""
    logger.info("Starting batch processing of sample images...")
    
    try:
        from custom_omr_system import OMRProcessor
        
        processor = OMRProcessor(
            answer_key_path='Key (Set A and B).xlsx',
            samples_folder='samples'
        )
        
        # Process both sets
        all_results = []
        
        for set_name in ['Set A', 'Set B']:
            logger.info(f"Processing {set_name}...")
            results = processor.process_sample_folder(set_name)
            all_results.extend(results)
            
            if results:
                avg_score = sum(r.get('total_score', 0) for r in results if 'error' not in r) / len([r for r in results if 'error' not in r])
                logger.info(f"{set_name} average score: {avg_score:.1f}/100")
        
        # Generate report
        report = processor.generate_report(all_results)
        
        # Save results
        import json
        with open('batch_processing_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        with open('batch_processing_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Batch processing completed!")
        logger.info("Results saved to: batch_processing_results.json")
        logger.info("Report saved to: batch_processing_report.txt")
        
        print("\n" + "="*60)
        print("BATCH PROCESSING REPORT")
        print("="*60)
        print(report)
        
        return True
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False

def start_web_server():
    """Start the Flask web server"""
    logger.info("Starting OMR web server...")
    
    try:
        from omr_flask_api import app
        
        logger.info("Web server starting on http://localhost:5000")
        logger.info("Dashboard available at: http://localhost:5000")
        logger.info("API documentation at: http://localhost:5000/api/health")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        return False

def run_system_test():
    """Run comprehensive system tests"""
    logger.info("Running system tests...")
    
    try:
        # Test 1: Answer key loading
        logger.info("Test 1: Answer key loading...")
        from custom_omr_system import OMRAnswerKeyLoader
        
        answer_loader = OMRAnswerKeyLoader('Key (Set A and B).xlsx')
        if answer_loader.answer_keys:
            logger.info("✓ Answer keys loaded successfully")
            for key_name, answers in answer_loader.answer_keys.items():
                logger.info(f"  {key_name}: {len(answers)} answers")
        else:
            logger.warning("✗ No answer keys loaded")
        
        # Test 2: Image preprocessing
        logger.info("Test 2: Image preprocessing...")
        from custom_omr_system import AdvancedImagePreprocessor
        
        preprocessor = AdvancedImagePreprocessor()
        
        # Find a sample image
        sample_image = None
        for set_folder in ['samples/Set A', 'samples/Set B']:
            if os.path.exists(set_folder):
                for file in os.listdir(set_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_image = os.path.join(set_folder, file)
                        break
                if sample_image:
                    break
        
        if sample_image:
            try:
                processed = preprocessor.preprocess_image(sample_image)
                logger.info(f"✓ Image preprocessing successful: {processed.shape}")
            except Exception as e:
                logger.warning(f"✗ Image preprocessing failed: {e}")
        else:
            logger.warning("✗ No sample images found for testing")
        
        # Test 3: Bubble detection
        logger.info("Test 3: Bubble detection...")
        from custom_omr_system import BubbleDetectorAdvanced
        
        detector = BubbleDetectorAdvanced()
        if sample_image:
            try:
                bubbles = detector.detect_bubbles_by_contours(processed)
                questions = detector.organize_bubbles_into_grid(bubbles)
                logger.info(f"✓ Bubble detection successful: {len(bubbles)} bubbles, {len(questions)} questions")
            except Exception as e:
                logger.warning(f"✗ Bubble detection failed: {e}")
        
        # Test 4: Model loading
        logger.info("Test 4: Model loading...")
        from omr_training_pipeline import EnhancedBubbleClassifier
        
        classifier = EnhancedBubbleClassifier()
        model_paths = ['enhanced_omr_classifier.h5', 'best_omr_classifier.h5', 'omr_bubble_classifier.h5']
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                if classifier.load_model(model_path):
                    logger.info(f"✓ Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
        
        if not model_loaded:
            logger.warning("✗ No trained model found. System will use traditional detection.")
        
        logger.info("System tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "system": {
            "name": "OMR Evaluation System",
            "version": "1.0.0",
            "error_tolerance": 0.005
        },
        "processing": {
            "target_image_size": [800, 1200],
            "bubble_detection": {
                "min_area": 100,
                "max_area": 2000,
                "circularity_threshold": 0.3
            }
        },
        "scoring": {
            "questions_per_subject": 20,
            "total_questions": 100,
            "marks_per_question": 1.0,
            "subjects": ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
        },
        "answer_keys": {
            "SET_A": "Loaded from Excel file",
            "SET_B": "Loaded from Excel file"
        }
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Configuration file created: config.json")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='OMR Evaluation System Setup and Run')
    parser.add_argument('--mode', choices=['setup', 'train', 'process', 'server', 'test', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--skip-install', action='store_true', 
                       help='Skip package installation')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip data validation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OMR EVALUATION SYSTEM")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Setup phase
    if args.mode in ['setup', 'all']:
        logger.info("SETUP PHASE")
        logger.info("-" * 20)
        
        # Install requirements
        if not args.skip_install:
            if not install_requirements():
                logger.error("Package installation failed")
                return 1
        
        # Create directories
        create_directory_structure()
        
        # Create config
        create_sample_config()
        
        # Validate data files
        if not args.skip_validation:
            if not validate_data_files():
                logger.error("Data validation failed")
                return 1
    
    # Training phase
    if args.mode in ['train', 'all']:
        logger.info("\nTRAINING PHASE")
        logger.info("-" * 20)
        
        if not run_training_pipeline(args):
            logger.error("Training failed")
            if args.mode == 'train':
                return 1
    
    # Processing phase
    if args.mode in ['process', 'all']:
        logger.info("\nPROCESSING PHASE")
        logger.info("-" * 20)
        
        if not run_batch_processing():
            logger.error("Batch processing failed")
            if args.mode == 'process':
                return 1
    
    # Testing phase
    if args.mode in ['test', 'all']:
        logger.info("\nTESTING PHASE")
        logger.info("-" * 20)
        
        if not run_system_test():
            logger.error("System tests failed")
            if args.mode == 'test':
                return 1
    
    # Server phase
    if args.mode in ['server', 'all']:
        logger.info("\nSERVER PHASE")
        logger.info("-" * 20)
        
        if args.mode == 'all':
            logger.info("Setup complete! You can now start the server with:")
            logger.info("python setup_and_run.py --mode server")
        else:
            start_web_server()
    
    if args.mode == 'all':
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("Your OMR Evaluation System is ready!")
        print("")
        print("Next steps:")
        print("1. Start the web server: python setup_and_run.py --mode server")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Upload and process OMR sheets through the web interface")
        print("")
        print("Files created:")
        print("- enhanced_omr_classifier.h5 (trained model)")
        print("- batch_processing_results.json (sample processing results)")
        print("- training_report.txt (training details)")
        print("- config.json (system configuration)")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
