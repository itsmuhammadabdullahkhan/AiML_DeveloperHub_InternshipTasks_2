🤖 Advanced Machine Learning Task Set
Author: Muhammad Abdullah Khan

Institution: Developer Hub & Co.

Position: Internship | March 2026

This repository contains the implementation of five specialized Machine Learning tasks, covering Natural Language Processing (NLP), Production-ready Pipelines, Multimodal Deep Learning, and LLM Prompt Engineering.

📂 Project Overview
Task 1: News Topic Classifier (BERT)
File: Task1_BERT_News.ipynb

Model: bert-base-uncased (Fine-tuned via Hugging Face Trainer).

Description: Fine-tuning a pre-trained BERT transformer to classify news headlines into four categories: World, Sports, Business, and Sci/Tech.

Key Features: Implementation of Transformer tokenization, GPU acceleration, and a Gradio web interface for real-time inference.

Task 2: End-to-End Production Pipeline (Churn Prediction)
File: Task2_Churn_Pipeline.ipynb

Model: Logistic Regression & Random Forest Classifier.

Description: A robust Scikit-learn Pipeline designed for production environments, automating the journey from raw data to prediction.

Key Features: Use of ColumnTransformer for automated One-Hot Encoding and Standard Scaling, and model persistence using joblib for deployment readiness.

Task 3: Image-Based Feature Extraction (CNN)
File: Task3_CNN_Features.ipynb

Architecture: 3-layer Convolutional Neural Network.

Description: Focused on deep learning computer vision to extract spatial features from image datasets, optimizing for high-dimensional visual data.

Key Features: Implementation of Max-Pooling, Dropout for regularization, and ReLU activation functions to handle non-linear image data.

Task 4: Multimodal Housing Price Prediction (Late Fusion)
File: Task4_Multimodal_Housing.ipynb

Architecture: Dual-Branch Neural Network (CNN + MLP).

Description: An advanced regression model that "fuses" visual data (house images) and numerical data (property features) to predict prices.

Key Features: * CNN Branch: Extracts visual features from property images.

MLP Branch: Processes numerical housing data (area, bedrooms, age).

Fusion: Feature concatenation followed by a final regression head, evaluated using MAE and RMSE.

Task 5: Auto-Tagging Support Tickets (LLM vs. Fine-Tuning)
File: Task5_LLM_Tagging.ipynb

Models: DistilBERT (Fine-tuned) vs. Llama-3.2 / FLAN-T5 (via HF Router).

Description: A comparative study between traditional fine-tuning and modern Large Language Model (LLM) prompting for ticket classification.

Key Features: * Fine-Tuning: Training a specialized DistilBERT classifier on 200+ samples.

Zero/Few-Shot: Utilizing the 2026 Hugging Face Router API for real-time LLM inference.

Comparison: Analysis of accuracy vs. reasoning capabilities in automated tagging (Technical, Billing, Account).

🖥️ Technical Note: Infrastructure & Optimization
During development, the project was migrated from a local Jupyter environment to Google Colab due to high computational demands:

Hardware Optimization: BERT-base and Multimodal CNN architectures require significant VRAM. Local execution led to kernel crashes; Colab's NVIDIA T4 GPU provided the necessary stability.

Execution Strategy: Implemented Parallel Computing and Mixed Precision (FP16) to ensure efficient processing of deep learning workloads.

Scalability: This shift demonstrates a "Cloud-First" approach, utilizing industry-standard remote computing resources.

🛠️ Setup & Requirements
To run these notebooks, ensure you have the following installed:

Bash
pip install torch torchvision tensorflow transformers datasets scikit-learn gradio joblib pandas numpy
How to Run:

Upload the .ipynb files to Google Colab.

Enable the GPU (Runtime > Change runtime type > T4 GPU).

For Task 5, ensure your HF_TOKEN is added to your Colab Secrets.

Execute the cells sequentially to see the training logs and final evaluation metrics.

📊 Submission Verification
All notebooks include saved execution outputs, loss curves, and final evaluation metrics (Accuracy, MAE, and RMSE) as per the project requirements.
