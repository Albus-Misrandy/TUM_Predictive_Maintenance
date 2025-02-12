# TUM Predictive Maintenance

## Project Overview

This project aims to develop and implement a predictive maintenance model to estimate the **Remaining Useful Life (RUL)** of equipment, improving maintenance efficiency and reducing unexpected downtime. The project is part of a research initiative or coursework at **Technical University of Munich (TUM)**.

## Features

- **Data Preprocessing**: Handles the **C-MAPSS dataset**, including feature engineering and data cleaning.
- **Model Training**: Uses **deep learning models (e.g., LSTM)** to predict RUL.
- **Results Visualization**: Provides visualization tools for analyzing model performance.

## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- Required dependencies listed in `requirements.txt`

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Albus-Misrandy/TUM_Predictive_Maintenance.git
   cd TUM_Predictive_Maintenance
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   conda create --name predictive_maintenance python=3.11
   conda activate predictive_maintenance

   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Run the following command to start model training:

```bash
pthon main.py --train_data_path ".CMAPSSdata/train_FD001.txt" --test_data_path "./CMAPSSdata/test_FD001.txt" --rul_data_path "./CMAPSSdata/RUL_FD001.txt" --num_epochs your_num ----regression_model_save_path "Your model save path" --classification_model_save_path "Your path"
```

The predicted results have been saved in the `predicted_data.csv` file.

## Result

All the results are shown in Image Folders:
For example:
CNN_LSTM predicted error
<img src="https://github.com/Albus-Misrandy/TUM_Predictive_Maintenance/blob/main/Image/CNN_LSTM_Regression_Error.png" width="500">

CNN_Transformer predicted error
<img src="https://github.com/Albus-Misrandy/TUM_Predictive_Maintenance/blob/main/Image/CNN_Transformer_Regression_Error.png" width="500">
## Project Structure

```
TUM_Predictive_Maintenance/
├── Environment               # Environment 
├── Articles/               # Related research articles or documents
├── CMAPSSdata/             # Raw dataset
├── Image/                  # Image resources
├── Models/                 # Saved models
├── __pycache__/            # Python cache files
├── README.md               # Project documentation
├── main.py                 # Main script
├── predicted_data.csv      # Predicted results
└── utils.py                # Utility functions
```

## Dependencies & Tech Stack

- **Python 3.x**
- **PyTorch** 
- **Pandas**
- **NumPy**
- **Matplotlib**

Make sure to check `requirements.txt` for the full list of dependencies.

## Contribution Guide

We welcome contributions! Please follow these steps:

1. **Fork** the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Author & Contact

- **Author**: Albus-Misrandy
- **Contact**: Please raise an issue on GitHub for any questions or suggestions.

## Acknowledgments

- Special thanks to **TUM** for supporting this research.
- Thanks to all contributors who have helped improve this project.
