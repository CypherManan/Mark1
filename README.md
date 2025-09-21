# *Mark1: Automated OMR Evaluation & Scoring System*

## *Project Overview*
Welcome to Mark1, an *automated OMR (Optical Mark Recognition) evaluation and scoring system* designed for the *Code4Edtech* hackathon hosted by Innomatics Research Labs. Our objective is to create a scalable and highly accurate solution that can evaluate OMR sheets captured with a mobile phone camera, significantly reducing the time and effort involved in manual grading. This system not only provides quick results but also helps educators focus on providing insights and engaging with students rather than getting bogged down by the logistics of evaluation.


## *Key Features*
* *Mobile-Friendly Capture*: Evaluate OMR sheets captured using a standard mobile phone camera.
* *Robust Preprocessing: Corrects for common image distortions like **rotation, **skew, and uneven **illumination* to ensure accurate bubble detection.
* *High Accuracy Bubble Detection: Employs a hybrid approach using **classical Computer Vision techniques (OpenCV)* for efficiency, with *ML-based classifiers* to handle ambiguous or poorly marked bubbles. Our target is a sub-0.5% error rate.
* *Multi-Version Support*: Handles multiple versions or sets of the same exam (up to four sets) to prevent cheating and ensure fair evaluation.
* *Web Application Interface*: A user-friendly web application allows evaluators to upload sheets, monitor the evaluation process, and manage results.
* *Detailed Scoring: Provides **per-subject scores* (0-20 for each of the five subjects) and a *total score* (0-100).
* *Data Audit Trail*: Stores rectified sheet images and a transparent overlay of the marked bubbles, along with a JSON-formatted result file for easy auditing and verification.
* *Fast Turnaround*: Reduces evaluation time from days to minutes, revolutionizing the grading process.

---

## *Proposed Solution & Workflow*
Our system is built around a comprehensive and automated workflow. 
1.  *Student fills OMR sheet* during the examination.
2.  *Digitization*: An evaluator uses a mobile phone to capture a clear image of each OMR sheet.
3.  *Upload*: The evaluator uploads the images to our web application.
4.  *Automated Pipeline*: The system's backend processes the images through a series of steps:
    * *Orientation Detection*: Identifies the correct orientation of the OMR sheet.
    * *Perspective Rectification*: Corrects for any perspective distortion from the camera angle.
    * *Bubble Grid Identification*: Locates the grid of bubbles and extracts student responses.
    * *Classification*: Determines whether each bubble is marked or unmarked.
    * *Answer Key Matching*: Compares the extracted answers with the correct answer key, based on the sheet version.
    * *Scoring*: Calculates section-wise and total scores.
5.  *Result Generation: The scores are stored in a secure database and can be exported as **CSV* or *Excel* files.
6.  *Evaluator Dashboard*: The web application features a dashboard where evaluators can view student summaries and aggregate statistics.

---

## *Technical Stack*
* *Frontend*: (e.g., React, Vue.js, or similar) for the web application interface.
* *Backend*: (e.g., Python with Flask/Django) to manage the API and image processing pipeline.
* *Computer Vision: **OpenCV* for classical image processing tasks.
* *Machine Learning*: (e.g., Scikit-learn, TensorFlow, or PyTorch) for the classifier used to resolve ambiguous cases.
* *Database*: (e.g., PostgreSQL, MongoDB) for secure storage of student results and audit data.
* *Deployment*: (e.g., Docker, Heroku, AWS) for a scalable and reliable online service.

---

## *Team Mark1*
    * Arnav Nigam*: Project Manager (ML Engineer)
    * *Aditya Kumar:  Frontend Developer    * **Manan Mittal*:  Backend Developer (CNN Pipeline Manager)
