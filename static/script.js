// Start the video stream when the page loads
window.addEventListener("load", async () => {
    let video = document.getElementById("camera-stream");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing the camera:", error);
    }
});

document.getElementById("inference-btn").addEventListener("click", async function() {
    let video = document.getElementById("camera-stream");
    let canvas = document.getElementById("capture-canvas");
    let context = canvas.getContext('2d');

    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current frame from the video onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a Blob
    canvas.toBlob(async (blob) => {
        // Create a new FormData instance
        var formData = new FormData();

        // Append the image file to the FormData instance
        formData.append("image", blob, "snapshot.png");

        // Send the image file to the server using fetch API

        try {
            let response = await fetch('http://127.0.0.1:8000/inference/', {
                method: 'POST',
                body: formData
            });
            let responseData = await response.text(); // Get the response as text

            // Check if responseData is not empty and is a valid JSON string
            if (responseData) {
                try {
                    // Parse the text to JSON
                    responseData = JSON.parse(responseData);
                    console.log("Parsed responseData:", responseData);
                    console.log("Parsed responseData:", responseData[0]);
                    console.log( typeof  responseData)

                    // Display the results on the screen
                    displayResults(responseData);

                    // Post the results to the "recommend" API
                    postRecommendationRequest(responseData);
                } catch (e) {
                    console.error("Error parsing JSON:", e);
                }
            }
        } catch (error) {
            console.error("Error posting the image:", error);
        }
    }, 'image/png');
});

function displayResults(data) {
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = '';

    // Accessing the first object of the array
    const infoObj = data[0];
    console.log("infoObj:", infoObj);

    // Display the image
    const img = document.createElement("img");
    img.src = `http://127.0.0.1:8000/${infoObj.file_name}`;
    img.alt = "Inference result";
    resultsDiv.appendChild(img);

    // Display the information
    const info = document.createElement("p");
    info.textContent = `Race: ${infoObj.race}, Age: ${infoObj.age}, Emotion: ${infoObj.emotion}, Gender: ${infoObj.gender}, Skin Tone: ${infoObj.skintone}, Masked: ${infoObj.masked}`;
    resultsDiv.appendChild(info);
}


async function postRecommendationRequest(data) {
    // Accessing the first object of the array
    const attributes = {
        race: data[0].race,
        age: data[0].age,
        emotion: data[0].emotion,
        gender: data[0].gender,
        skintone: data[0].skintone,
        masked: data[0].masked
    };
    console.log("Attributes to send to /recommend/:", attributes);

    try {
        const response = await fetch('http://127.0.0.1:8000/recommend/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(attributes)
        });
        const recommendationData = await response.json();
        displayRecommendations(recommendationData);
    } catch (error) {
        console.error("Error posting the recommendation request:", error);
    }
}


function displayRecommendations(data) {
    const recommendationsDiv = document.getElementById("recommendations");
    recommendationsDiv.innerHTML = '';

    data.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.textContent = `Item: ${item.name}, Size: ${item.size}, Color: ${item.color}, Material: ${item.material}`;
        recommendationsDiv.appendChild(itemDiv);
    });
}
