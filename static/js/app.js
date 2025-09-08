document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const resultsContainer = document.getElementById('results');
    const loadingOverlay = document.getElementById('loadingOverlay');

    const API_ENDPOINT = 'http://localhost:5000';
    let selectedFiles = new Set();

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#4CAF50';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ddd';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ddd';
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        const validFiles = Array.from(files).filter(file => {
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
            return validTypes.includes(file.type);
        });

        validFiles.forEach(file => {
            if (!Array.from(selectedFiles).some(f => f.name === file.name)) {
                selectedFiles.add(file);
                displayPreview(file);
            }
        });

        updatePredictButton();
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <button class="remove-btn" data-filename="${file.name}">&times;</button>
            `;
            imagePreview.appendChild(previewItem);

            previewItem.querySelector('.remove-btn').addEventListener('click', () => {
                selectedFiles.delete(file);
                previewItem.remove();
                updatePredictButton();
            });
        };
        reader.readAsDataURL(file);
    }

    function updatePredictButton() {
        predictBtn.disabled = selectedFiles.size === 0;
    }

    predictBtn.addEventListener('click', async () => {
        if (selectedFiles.size === 0) return;

        loadingOverlay.style.display = 'flex';
        resultsContainer.innerHTML = '';

        try {
            const predictions = await predictImages(Array.from(selectedFiles));
            displayResults(predictions);
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Error processing images. Please try again.');
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });

    async function predictImages(files) {
        const formData = new FormData();
        
        if (files.length === 1) {
            // Single image prediction
            formData.append('file', files[0]);
            const response = await fetch(`${API_ENDPOINT}/predict`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Prediction failed');
            const result = await response.json();
            return [{ ...result, filename: files[0].name }];
        } else {
            // Batch prediction
            files.forEach(file => {
                formData.append('files[]', file);
            });
            
            const response = await fetch(`${API_ENDPOINT}/predict/batch`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Batch prediction failed');
            const result = await response.json();
            return result.results;
        }
    }

    function displayResults(predictions) {
        predictions.forEach(prediction => {
            if (prediction.error) {
                // Handle error case
                const errorCard = document.createElement('div');
                errorCard.className = 'result-card error';
                errorCard.innerHTML = `
                    <h3>Error: ${prediction.filename}</h3>
                    <p>${prediction.error}</p>
                `;
                resultsContainer.appendChild(errorCard);
                return;
            }

            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';

            // Find the original file for preview
            const file = Array.from(selectedFiles).find(f => f.name === prediction.filename);
            const imageUrl = file ? URL.createObjectURL(file) : '';

            const probabilities = prediction.probabilities;
            const probabilitiesHtml = Object.entries(probabilities)
                .map(([food, probability]) => `
                    <div class="probability-item">
                        <div class="probability-label">
                            <span>${food}</span>
                            <span>${(probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${probability * 100}%"></div>
                        </div>
                    </div>
                `).join('');

            resultCard.innerHTML = `
                <img src="${imageUrl}" alt="${prediction.filename}">
                <div class="result-info">
                    <h3>${prediction.predicted_food}</h3>
                    <p>Confidence Scores:</p>
                    ${probabilitiesHtml}
                </div>
            `;

            resultsContainer.appendChild(resultCard);
        });
    }

    // Check API health on load
    fetch(`${API_ENDPOINT}/health`)
        .catch(error => {
            console.error('API health check failed:', error);
            alert('Warning: API server might be unavailable. Please ensure the server is running.');
        });
});
