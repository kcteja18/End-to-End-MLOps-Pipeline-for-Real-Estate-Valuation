# End-to-End MLOps Pipeline for Real Estate Valuation

This project implements a complete MLOps pipeline for predicting California housing prices. It features automated training, evaluation, and deployment of a machine learning model through a REST API and web interface.

## Project Structure

```
├── app.py                 # Streamlit web interface
├── src/
│   ├── main.py           # FastAPI server implementation
│   ├── process_data.py   # Data preprocessing script
│   └── train.py          # Model training and MLflow logging
├── data/
│   └── processed/        # Processed training and test datasets
├── Dockerfile            # Container configuration
├── requirements.txt      # Project dependencies
└── .github/workflows/    # CI/CD pipeline configurations
```

## Features

- Data preprocessing pipeline for California Housing dataset
- RandomForest model training with MLflow experiment tracking
- FastAPI REST API for model serving
- Streamlit web interface for predictions
- Containerized deployment with Docker
- Automated CI/CD pipeline with GitHub Actions

## Getting Started

### Prerequisites

- Python 3.11
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/End-to-End-MLOps-Pipeline-for-Real-Estate-Valuation.git
cd End-to-End-MLOps-Pipeline-for-Real-Estate-Valuation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

1. Process the data:
```bash
python src/process_data.py
```

2. Train the model:
```bash
python src/train.py
```

3. Start the FastAPI server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

4. Launch the Streamlit interface:
```bash
streamlit run app.py
```

### Using Docker

Build and run the container:
```bash
docker build -t real-estate-predictor .
docker run -p 80:80 real-estate-predictor
```

## API Endpoints

- `GET /`: Welcome message
- `POST /predict/`: Get house price prediction

Example prediction request:
```json
{
    "MedInc": 8.3,
    "HouseAge": 41.0,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 560,
    "AveOccup": 3.0,
    "Latitude": 34.2,
    "Longitude": -118.5
}
```

## Model Tracking

The project uses MLflow for experiment tracking. Models and metrics are stored in:
- `mlruns/`: MLflow tracking data
- `mlartifacts/`: Model artifacts

## CI/CD Pipeline

The GitHub Actions workflow (`main.yml`) automates:
1. Data processing
2. Model training
3. Docker image building
4. Container registry pushing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
