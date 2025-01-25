// Global variables
var PointCalibrate = 0;
var CalibrationPoints = {};
var helpModal;

// Constants
const CALIBRATION_TITLE = "Calibration";
const ACCURACY_LABEL = "Accuracy";
const NOT_CALIBRATED_LABEL = "Not yet Calibrated";

/**
 * Clear the canvas and the calibration buttons.
 */
function ClearCanvas() {
    document.querySelectorAll(".Calibration").forEach((i) => {
        i.style.setProperty("display", "none");
    });
    const canvas = document.getElementById("plotting_canvas");
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Show the instruction popup for calibration at startup.
 */
function PopUpInstruction() {
    ClearCanvas();
    Swal.fire({
        title: CALIBRATION_TITLE,
        text: "Please click on each of the 9 points on the screen. You must click on each point 5 times until it turns yellow.",
        confirmButtonText: "Start"
    }).then((result) => {
        if (result.isConfirmed) {
            ShowCalibrationPoint();
        }
    });
}

/**
 * Show the help modal instructions.
 */
function helpModalShow() {
    if (!helpModal) {
        helpModal = new bootstrap.Modal(document.getElementById("helpModal"));
    }
    helpModal.show();
}

/**
 * Calculate and display accuracy after calibration.
 */
function calcAccuracy() {
    Swal.fire({
        title: "Calculating measurement",
        text: "Please don't move your mouse & stare at the middle dot for 5 seconds.",
        allowOutsideClick: false,
        didOpen: () => {
            store_points_variable(); // Start storing points
        }
    }).then(() => {
        sleep(5000).then(() => {
            stop_storing_points_variable(); // Stop storing points
            try {
                const past50 = webgazer.getStoredPoints();
                const precisionMeasurement = calculatePrecision(past50);
                const accuracyLabel = `<a>${ACCURACY_LABEL} | ${precisionMeasurement}%</a>`;
                document.getElementById("Accuracy").innerHTML = accuracyLabel;

                Swal.fire({
                    title: `Your accuracy measure is ${precisionMeasurement}%`,
                    showCancelButton: true,
                    confirmButtonText: "Continue",
                    cancelButtonText: "Recalibrate"
                }).then((result) => {
                    if (!result.isConfirmed) {
                        webgazer.clearData();
                        ClearCalibration();
                        ClearCanvas();
                        ShowCalibrationPoint();
                    } else {
                        ClearCanvas();
                    }
                });
            } catch (error) {
                console.error("Error during precision calculation:", error);
                Swal.fire("Error", "An error occurred during precision calculation.", "error");
            }
        });
    });
}

/**
 * Handle click on calibration points.
 */
function calPointClick(node) {
    const id = node.id;

    if (!CalibrationPoints[id]) {
        CalibrationPoints[id] = 0; // Initialize point
    }
    CalibrationPoints[id]++;

    if (CalibrationPoints[id] === 5) {
        node.style.setProperty("background-color", "yellow");
        node.setAttribute("disabled", "disabled");
        PointCalibrate++;
    } else if (CalibrationPoints[id] < 5) {
        const opacity = 0.2 * CalibrationPoints[id] + 0.2;
        node.style.setProperty("opacity", opacity);
    }

    if (PointCalibrate === 8) {
        document.getElementById("Pt5").style.removeProperty("display");
    }

    if (PointCalibrate >= 9) {
        document.querySelectorAll(".Calibration").forEach((i) => {
            i.style.setProperty("display", "none");
        });
        document.getElementById("Pt5").style.removeProperty("display");

        ClearCanvas(); // Clear the canvas
        calcAccuracy(); // Calculate accuracy
    }
}

/**
 * Load this function when the page starts.
 */
function docLoad() {
    ClearCanvas();
    helpModalShow();

    document.querySelectorAll(".Calibration").forEach((i) => {
        i.addEventListener("click", () => calPointClick(i));
    });
}
window.addEventListener("load", docLoad);

/**
 * Show the Calibration Points.
 */
function ShowCalibrationPoint() {
    document.querySelectorAll(".Calibration").forEach((i) => {
        i.style.removeProperty("display");
    });
    document.getElementById("Pt5").style.setProperty("display", "none");
}

/**
 * Clear calibration memory.
 */
function ClearCalibration() {
    document.querySelectorAll(".Calibration").forEach((i) => {
        i.style.setProperty("background-color", "red");
        i.style.setProperty("opacity", "0.2");
        i.removeAttribute("disabled");
    });

    CalibrationPoints = {};
    PointCalibrate = 0;
}

/**
 * Sleep function to delay execution.
 */
function sleep(time) {
    return new Promise((resolve) => setTimeout(resolve, time));
}