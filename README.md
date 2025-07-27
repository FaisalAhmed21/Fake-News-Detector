# Fake News Detector

A web-based application that uses machine learning to detect fake news articles. Users can paste any news article and get instant analysis with confidence scores.

## Features

- 🔍 **Real-time Analysis**: Paste any news article and get instant results
- 📊 **Confidence Scoring**: See how confident the model is in its prediction
- 🎨 **Modern UI**: Beautiful, responsive design that works on all devices
- 💡 **Example Texts**: Try pre-loaded examples to test the system
- 📱 **Mobile Friendly**: Optimized for desktop and mobile devices

## How It Works

The application uses a machine learning model trained on a dataset of real and fake news articles. The model:

1. **Preprocesses** the input text (removes special characters, converts to lowercase)
2. **Vectorizes** the text using TF-IDF (Term Frequency-Inverse Document Frequency)
3. **Predicts** whether the news is real or fake using Logistic Regression
4. **Returns** the result with a confidence score

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

### Step 3: Open in Browser
Open your web browser and go to: `http://localhost:5000`

## Usage

1. **Paste News Article**: Copy and paste any news article into the text area
2. **Click Analyze**: Press the "🔍 Analyze News" button
3. **View Results**: See if the news is classified as REAL or FAKE with confidence score
4. **Try Examples**: Use the example buttons to test with pre-loaded samples

## Dataset Information

The application uses a dataset of news articles to train the machine learning model. The dataset is available in the following Google Drive folder:

**Dataset Source**: [Google Drive - Fake News Dataset](https://drive.google.com/drive/folders/1ByadNwMrPyds53cA6SDCHLelTAvIdoF_)

The dataset contains two CSV files:
- **Fake.csv** (59.9 MB) - Contains fake news articles
- **True.csv** (51.1 MB) - Contains real news articles

## Model Information

- **Algorithm**: Logistic Regression
- **Training Data**: 44,898 news articles (mix of real and fake)
- **Features**: TF-IDF vectorization of preprocessed text
- **Accuracy**: ~98.8% on test data

## File Structure

```
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Frontend HTML/CSS/JavaScript
├── Fake.csv              # Fake news dataset
├── True.csv              # Real news dataset
├── requirements.txt       # Python dependencies
└── README.md            # This file
```

## API Endpoints

- `GET /` - Main application page
- `POST /predict` - Predict if news is fake or real
  - Request: `{"text": "news article text"}`
  - Response: `{"result": "REAL/FAKE", "confidence": 95.5, "prediction": 1}`

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Machine Learning**: scikit-learn, pandas, numpy
- **Text Processing**: NLTK, regular expressions

## Contributing

Feel free to contribute to this project by:
- Improving the model accuracy
- Adding new features
- Enhancing the UI/UX
- Reporting bugs

## License

This project is open source and available under the MIT License. 
