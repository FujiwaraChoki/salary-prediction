# Salary Prediction and Visualization Project

This project aims to predict future salaries based on age using a neural network model and visualize the results. It also includes data preprocessing, model training, and data visualization steps.

## Project Structure

- `data_processing.py`: Contains functions to load and preprocess the employee data from a CSV file.
- `visualization.py`: Includes functions to visualize gender and department distributions.
- `clustering.py`: Contains code for performing clustering analysis (not shown in the provided code).
- `main.py`: The main script that brings everything together. It loads the data, visualizes distributions, trains the salary prediction model, and visualizes predicted future salaries.

## Getting Started

1. Clone this repository:

    ```
    git clone https://github.com/fujiwarachoki/salary-prediction.git
    cd salary-prediction-project
    ```

2. Install the required packages:
    
    ```
    pip install -r requirements.txt
    ```

3. Run the main script:

    ```
    cd src
    python3 main.py
    ```

## Usage

- The main script, `main.py`, loads and processes the data, performs visualizations, trains the salary prediction model, and visualizes predicted future salaries.
- You can customize the neural network model architecture and training parameters in `main.py`.
- Additional functionalities can be added by extending the existing files or adding new ones.

## Acknowledgments

- This project is inspired by the need to predict future salaries based on age using machine learning techniques.
- The project structure and README template are adapted from common practices in software development.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.