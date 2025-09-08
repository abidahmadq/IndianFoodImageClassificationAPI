# Indian Food Image Classification API

A web application that classifies Indian food images using a deep learning model. The application includes both a REST API backend and a modern web frontend.

## Features

- Image classification for Indian food items (Poha, SabudanaKhichdi, Samosa)
- Modern, responsive web interface
- Drag-and-drop file upload
- Multiple image batch processing
- Real-time prediction results with confidence scores
- RESTful API endpoints

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd IndianFoodImageClassificationAPI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the model file is present:
- Make sure `projectmodel.h5` is in the root directory

## Running the Application

1. Start the Flask server:
```bash
python run.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

### 1. Health Check
- **URL:** `/health`
- **Method:** GET
- **Response:** Server health status and timestamp

### 2. Model Information
- **URL:** `/model/info`
- **Method:** GET
- **Response:** Model classes and configuration

### 3. Single Image Prediction
- **URL:** `/predict`
- **Method:** POST
- **Body:** Form-data with 'file' field
- **Response:** Prediction results with confidence scores

### 4. Batch Prediction
- **URL:** `/predict/batch`
- **Method:** POST
- **Body:** Form-data with 'files[]' field (multiple files)
- **Response:** Array of prediction results

## Directory Structure

```
IndianFoodImageClassificationAPI/
├── run.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── projectmodel.h5        # Trained model file
├── static/
│   ├── css/
│   │   └── style.css     # Application styles
│   ├── js/
│   │   └── app.js        # Frontend JavaScript
│   └── images/
│       └── upload-icon.svg
├── templates/
│   └── index.html        # Main HTML template
└── uploads/              # Temporary image upload directory
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- Missing files
- Server errors
- Model prediction errors

## Browser Compatibility

The web interface is compatible with:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.