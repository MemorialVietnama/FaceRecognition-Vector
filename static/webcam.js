const video = document.getElementById('webcam-video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture-button');
const resultParagraph = document.getElementById('result');

// Запуск вебкамеры
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        console.log("Доступ к вебкамере получен");
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Ошибка при доступе к вебкамере:", err);
        alert(`Ошибка: ${err.message}`);
    });

// Кнопка захвата изображения
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Преобразуем изображение в Base64
    const imageData = canvas.toDataURL('image/jpeg');

    // Отправляем изображение на сервер
    fetch('/capture_and_process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: imageData })
    })
    .then(response => response.json())
    .then(data => {
        resultParagraph.textContent = data.message;
    })
    .catch(error => {
        console.error("Ошибка при обработке изображения:", error);
    });
});