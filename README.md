Advanced Machine Learning Task Set
Author: Muhammad Abdullah Khan

Institution: Developer Hub and Co.

Position: Internship

Date: March 2026

This repository contains the implementation of three specialized Machine Learning tasks, covering Natural Language Processing (NLP), Production-ready Pipelines, and Multimodal Deep Learning.

📂 Project Overview
Task 1: News Topic Classifier (BERT)
File: Task1_BERT_News.ipynb

Model: bert-base-uncased (Fine-tuned via Hugging Face Trainer).

Description: Fine-tuning a pre-trained BERT transformer to classify news headlines into four categories: World, Sports, Business, and Sci/Tech.

Key Features: * Implementation of Transformer tokenization (padding/truncation).

Training with GPU acceleration.

Real-time inference capability using a Gradio web interface.

Task 2: End-to-End Production Pipeline (Churn Prediction)
File: Task2_Churn_Pipeline.ipynb

Model: Logistic Regression & Random Forest Classifier.

Description: A robust Scikit-learn Pipeline designed for production environments, automating the journey from raw data to prediction.

Key Features: * Use of ColumnTransformer for automated One-Hot Encoding and Standard Scaling.

Integration of preprocessing and modeling into a single Pipeline object to prevent data leakage.

Demonstrates model persistence using joblib for deployment readiness.

Task 3: Multimodal Housing Price Prediction (CNN + MLP)
File: Task3_Multimodal_Housing.ipynb

Architecture: Late Fusion Neural Network (PyTorch).

Description: An advanced regression model that "fuses" visual data (house images) and numerical data (property features) to predict prices.

Key Features: * CNN Branch: A 3-layer Convolutional Neural Network designed to extract visual features from images.

MLP Branch: A Multi-Layer Perceptron to process numerical housing data.

Fusion: Feature concatenation followed by a final regression head, evaluated using MAE and RMSE.

🖥️ Technical Note: Migration to Cloud Infrastructure
During development, the project was migrated from a local Jupyter environment to Google Colab due to the high computational demands of the models:

Hardware Optimization: BERT-base and CNN architectures require significant VRAM and System RAM. Local execution on standard consumer hardware led to kernel crashes and memory overflow.

GPU Acceleration: By utilizing Google Colab’s NVIDIA T4 GPU, I implemented Parallel Computing and Mixed Precision (FP16), ensuring stable training and efficient processing that was not possible on a local CPU.

Scalability: This shift demonstrates a "Cloud-First" approach, utilizing industry-standard remote computing resources for heavy Deep Learning workloads.

🛠️ Setup & Requirements
To run these notebooks, ensure you have the following installed:

Bash
pip install torch torchvision transformers datasets scikit-learn gradio joblib pandas numpy
How to Run:
Upload the .ipynb files to Google Colab.

Enable the GPU (Runtime > Change runtime type > T4 GPU).

Execute the cells sequentially to see the training logs and final evaluation metrics.

Submission Verification
All notebooks include saved execution outputs, loss curves, and final evaluation metrics (Accuracy, MAE, and RMSE) as per the project requirements.
