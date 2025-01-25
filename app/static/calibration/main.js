console.log("main.js loaded");

window.addEventListener("load", async function () {
    console.log("Initializing WebGazer...");

    try {
        // Start the WebGazer tracker
        await webgazer
            .setRegression('ridge') // Required for initialization
            //.setTracker('clmtrackr') // Optional tracker (default is ridge)
            .setGazeListener(function (data, clock) {
                
            })
            .saveDataAcrossSessions(true) // Save across sessions
            .begin(); // Start tracking

        // Configure WebGazer's behavior
        webgazer
            .showVideoPreview(true) // Show the video preview
            .showPredictionPoints(true) // Show prediction points on screen
            .applyKalmanFilter(true); // Use Kalman filter for smoother predictions

        console.log("WebGazer started successfully.");
    } catch (error) {
        console.error("Error initializing WebGazer:", error);
    }

    setupCanvas(); // Set up the calibration canvas
    Restart(); // Begin the calibration process
});

/**
 * Set up the calibration canvas
 */
function setupCanvas() {
    const canvas = document.getElementById("plotting_canvas");
    if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.position = 'fixed';
        console.log("Canvas setup complete.");
    } else {
        console.error("Canvas element not found!");
    }
}

// Ensure data persists across sessions
window.saveDataAcrossSessions = true;

/**
 * End WebGazer tracking when the page is unloaded
 */
window.onbeforeunload = function () {
    console.log("Ending WebGazer session...");
    webgazer.end(); // Stop WebGazer
};

/**
 * Restart the calibration process
 * Clears stored data, resets calibration points, and shows instructions
 */
function Restart() {
    console.log("Restarting calibration process...");
    
    // Reset accuracy display
    const accuracyElement = document.getElementById("Accuracy");
    if (accuracyElement) {
        accuracyElement.innerHTML = "<a>Not yet Calibrated</a>";
    } else {
        console.error("Accuracy element not found!");
    }

    // Clear WebGazer's stored data and reset calibration
    webgazer.clearData();
    ClearCalibration();
    PopUpInstruction();
}
