document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    let form = e.target;
    let formData = new FormData(form);

    let response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    let result = await response.blob();
    let imgUrl = URL.createObjectURL(result);
    document.getElementById('denoisedImage').src = imgUrl;
});
