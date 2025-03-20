document.addEventListener('DOMContentLoaded', function() {
    // Update range input displays
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const display = input.nextElementSibling;
        
        // Initial value
        display.textContent = input.value;
        
        // Update on change
        input.addEventListener('input', () => {
            display.textContent = input.value;
        });
    });

    // Form submission
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('result');
    const predictionValue = document.getElementById('predictionValue');
    const predictionMessage = document.getElementById('predictionMessage');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Tahmin Ediliyor...';
        submitButton.disabled = true;

        try {
            const formData = new FormData(form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Update UI with prediction
                resultContainer.classList.remove('hidden');
                predictionValue.textContent = data.prediction;
                
                // Add message based on prediction score
                let message = '';
                if (data.prediction >= 80) {
                    message = 'Bu müzik hit olma potansiyeline sahip! 🌟';
                } else if (data.prediction >= 60) {
                    message = 'Bu müzik oldukça popüler olabilir! 🎵';
                } else if (data.prediction >= 40) {
                    message = 'Bu müzik ortalama bir popülerliğe sahip olabilir. 📊';
                } else {
                    message = 'Bu müzik niş bir dinleyici kitlesine hitap edebilir. 🎯';
                }
                predictionMessage.textContent = message;
            } else {
                throw new Error(data.error || 'Tahmin yapılırken bir hata oluştu.');
            }
        } catch (error) {
            alert('Hata: ' + error.message);
        } finally {
            // Restore button state
            submitButton.textContent = originalButtonText;
            submitButton.disabled = false;
        }
    });
}); 