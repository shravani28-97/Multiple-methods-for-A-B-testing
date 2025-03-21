**AI-Powered Causal Inference for Pricing A/B Testing**

Welcome to the AI-Powered Causal Inference app for pricing A/B testing! This app provides three causal inference methods—Difference-in-Differences (DiD), Synthetic Control (Naive), and Propensity Score Matching (PSM)—together with AI-generated recommendations for actionable business insights.

**Table of Contents**

  Overview

  Features
  
  Installation
  
  Usage
  
  Causal Inference Methods
  
  AI Integration
  
  Project Structure
  
  Contributing
  
  Contact
  
  Overview

Why this project?

Pricing Strategy: Decide whether a new price point is driving more sales or whether it’s time to pivot.
Policy Changes: Understand the impact of a strategic or organizational change.
Marketing Campaigns: Estimate the true effect of your latest campaign on conversions or revenue.
This app is built with Streamlit, making it super easy to upload data, select a method, and interpret results with AI assistance. Big thanks to ABC, whose insights on causal inference laid the foundation for this work!

Features

Upload Your Own Data
Drag-and-drop CSV uploads with automatic date parsing (if columns labeled as date).
Multiple Causal Methods
Difference-in-Differences (DiD)
Synthetic Control (Naive)
Propensity Score Matching (PSM)
AI Recommendations
Generates easy-to-understand recommendations based on each method’s results.
Interactive Visuals
Displays key plots and metrics, making your analyses more intuitive.
Easy Configuration
Minimal changes required—just select appropriate columns like treatment, outcome, date, or unit.
Installation

Clone the Repository
git clone https://github.com/your-username/ai_causal_inference_app.git
cd ai_causal_inference_app
Create and Activate a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
Install Dependencies
pip install -r requirements.txt
The required libraries typically include:

pandas
numpy
statsmodels
streamlit
scikit-learn
matplotlib
openai
Set OpenAI API Key
Make sure you have an OpenAI API Key.
Store it in .streamlit/secrets.toml (or as an environment variable). For example:
[general]
OPENAI_API_KEY = "your_openai_api_key_here"
Usage

Run the Streamlit App
streamlit run ai_causal_inference_app.py

Upload Your Data

On the web interface, upload a CSV file containing your treatment, outcome, and any required columns for the chosen method.

Select a Method

Choose between DiD, Synthetic Control, or PSM in the dropdown.

View Results & AI Recommendations

The app will display model outputs, plots, and an AI-generated recommendation on whether the strategy seems effective.

Causal Inference Methods

Difference-in-Differences (DiD)
Compares pre- and post-treatment changes between a treated and control group.

Synthetic Control (Naive)
Constructs a “synthetic twin” of your treated unit from a pool of control units and compares post-intervention outcomes.

Propensity Score Matching (PSM)
Matches treated units to control units with similar covariates, reducing selection bias in observational data.
AI Integration

The AI recommendations are powered by OpenAI’s GPT-3.5 (or later).
After each method’s calculation, a concise prompt is sent to the model to provide non-technical and actionable guidance.
Project Structure

ai_causal_inference_app/

  ├── ai_causal_inference_app.py    # Main Streamlit application
  
  ├── requirements.txt              # Project dependencies
  
  ├── README.md                     # You're reading this file
  
  └── ... (other supporting files, if any)
Contributing

Fork the project.
Create a new branch for your feature or bug fix.
Commit and push your changes.
Create a Pull Request and tag a maintainer for review.
We welcome contributions for additional features, bug fixes, or performance improvements!
