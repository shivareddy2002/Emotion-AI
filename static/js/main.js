async function analyzeEmotion() {
    const text = document.getElementById('userInput').value;
    const btn = document.getElementById('btnAnalyze');
    const resultDiv = document.getElementById('resultContainer');
    const confidenceBar = document.getElementById('confidenceBar');
    const emotionLabel = document.getElementById('emotionLabel');
    const confidenceLabel = document.getElementById('confidenceLabel');

    let statusNode = document.getElementById('statusMessage');
    if (!statusNode) {
        statusNode = document.createElement('div');
        statusNode.id = 'statusMessage';
        statusNode.className = 'mt-2 small';
        resultDiv.parentNode.insertBefore(statusNode, resultDiv);
    }
    statusNode.innerText = '';

    if (!text.trim()) {
        alert('Please enter some text to analyze.');
        return;
    }

    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing AI...';
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok || !data.emotion || !data.confidence) {
            const errorMsg = data.error || 'Prediction failed. Please try again.';
            statusNode.className = 'mt-2 small text-danger';
            statusNode.innerText = errorMsg;
            return;
        }

        emotionLabel.innerText = data.emotion.toUpperCase();
        confidenceLabel.innerText = data.confidence;

        const confValue = parseFloat(data.confidence) || 0;
        confidenceBar.style.width = confValue + '%';
        confidenceBar.setAttribute('aria-valuenow', String(confValue));

        if (data.warning) {
            statusNode.className = 'mt-2 small text-info';
            statusNode.innerText = data.warning;
        } else {
            statusNode.className = 'mt-2 small text-success';
            statusNode.innerText = data.source === 'lite-model' ? 'Prediction completed ' : 'Prediction completed.';
        }

        resultDiv.classList.remove('d-none');
        resultDiv.classList.add('animate-up');
    } catch (error) {
        console.error('Error:', error);
        statusNode.className = 'mt-2 small text-danger';
        statusNode.innerText = 'Network error while analyzing text.';
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Analyze Sentiment';
        btn.disabled = false;
    }
}
