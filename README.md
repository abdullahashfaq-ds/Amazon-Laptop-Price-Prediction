# Amazon Laptop Price Prediction Tool

The Amazon Laptop Price Prediction Tool is a machine learning based application designed to predict laptop prices listed on Amazon. The tool integrates machine learning algorithms with a Streamlit interface, providing an interactive experience where users can input laptop features and get a predicted price.

## Features

- Regression based ML Models for predicting laptop prices with high accuracy.
- Data Preprocessing pipeline to handle missing values, categorical encoding, feature scaling, and transformation.
- Pretrained Models for price prediction without retraining.
- Streamlit based interface to easily input laptop specifications such as brand, RAM, CPU, and more.

## Installation and Setup

Follow these steps to set up and run the application:

1. **Clone the Repository**

    ```bash
    git clone git@github.com:abdullahashfaq-ds/Amazon-Laptop-Price-Prediction.git
    cd Amazon-Laptop-Price-Prediction
    ```

2. **Create and Activate a Virtual Environment**

    For Linux/Mac:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    For Windows:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application**

    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.
