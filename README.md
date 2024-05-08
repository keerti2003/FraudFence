# FraudFence App

FraudFence is an application designed to help detect fraudulent transactions based on user data. It utilizes machine learning models to predict the likelihood of fraud in transactions and offers insights into potentially fraudulent activities.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The FraudFence App aims to provide users with a tool for identifying fraudulent transactions by analyzing transaction data. By training machine learning models on historical transaction data, the app can predict the likelihood of fraud in new transactions, enabling users to take preventive measures and mitigate potential losses.

## Features

- **Model Training**: Users can upload their transaction data to train machine learning models for fraud detection.
- **Prediction**: The app allows users to upload transaction data for prediction, and it identifies potentially fraudulent transactions.
- **Visualization**: FraudFence provides visualizations of fraud and non-fraud transactions to help users understand the distribution of fraudulent activities.

## Installation

To run the FraudFence App locally, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/keerti2003/FraudFence.git
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

## Usage

1. Launch the application by running `streamlit run main.py`.
2. On the home page, you'll find two options: "Train Model" and "Predict."
3. **Train Model**: Upload a CSV file containing transaction data to train the machine learning models for fraud detection. Enter a filename for the trained model.
4. **Predict**: Upload a CSV file containing transaction data to make predictions using a pre-trained model. Select a trained model from the dropdown menu.
5. After submitting the data, the app will display predictions and visualizations of fraudulent and non-fraudulent transactions.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.
