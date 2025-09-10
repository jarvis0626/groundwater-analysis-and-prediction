document.addEventListener("DOMContentLoaded", function () {
    fetchStates();
    setupEventListeners();
});

function fetchStates() {
    fetch('/get_states')
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(states => {
            console.log("Fetched States:", states); // Debugging

            const stateSelect = document.getElementById('state');
            stateSelect.innerHTML = "";

            if (!Array.isArray(states) || states.length === 0) {
                console.warn("No states received!");
                stateSelect.innerHTML = '<option value="">No data available</option>';
                return;
            }

            states.forEach(state => {
                let option = document.createElement("option");
                option.value = state;
                option.textContent = state;
                stateSelect.appendChild(option);
            });
        })
        .catch(error => console.error("Error loading states:", error));
}

function setupEventListeners() {
    document.getElementById('predict-btn').addEventListener('click', function () {
        const state = document.getElementById('state').value;
        const predictBtn = document.getElementById('predict-btn');

        if (!state) {
            alert("⚠️ Please select a state before proceeding.");
            return;
        }

        predictBtn.textContent = "Loading...";
        predictBtn.disabled = true;

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: state }),
        })
        .then(response => {
            if (!response.ok) throw new Error(`Server Error: ${response.status}`);
            return response.json();
        })
        .then(data => {
            predictBtn.textContent = "Show Data";
            predictBtn.disabled = false;

            if (data.error) {
                alert(`⚠️ ${data.error}`);
                return;
            }

            if (!data.years || !data.quality) {
                alert("⚠️ No data available for this state.");
                return;
            }

            updateChart(data.years, data.quality);
            displayEvaluationMetrics(data.metrics, data.feature_importance, data.avg_features);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`❌ An error occurred: ${error.message}`);
            predictBtn.textContent = "Show Data";
            predictBtn.disabled = false;
        });
    });
}

let chart;
function updateChart(years, quality) {
    const ctx = document.getElementById('qualityChart').getContext('2d');

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Water Quality',
                data: quality,
                borderColor: 'blue',
                borderWidth: 2,
                fill: false,
                pointRadius: 5,
                pointBackgroundColor: function(ctx) {
                    return ctx.dataIndex >= years.length - 3 ? 'red' : 'blue';
                },
                borderDash: function(ctx) {
                    return ctx.dataIndex >= years.length - 3 ? [5, 5] : [];
                },
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 }
                },
                y: {
                    min: 0,
                    max: 5,
                    ticks: { stepSize: 1 }
                }
            },
            plugins: {
                legend: { display: true, position: 'top' }
            }
        }
    });
}

function displayEvaluationMetrics(metrics, feature_importance, avgFeatures) {
    console.log("Received avgFeatures:", avgFeatures); // Debugging

    // Updating Evaluation Metrics
    document.getElementById('rmse').textContent = metrics?.rmse?.toFixed(3) ?? "-";
    document.getElementById('mae').textContent = metrics?.mae?.toFixed(3) ?? "-";
    document.getElementById('r2').textContent = metrics?.r2?.toFixed(3) ?? "-";

    // Check if avgFeatures is valid
    if (!avgFeatures || Object.keys(avgFeatures).length === 0) {
        console.warn("No average feature values available!");
        return;
    }

    // Updating Average Feature Values
    document.getElementById('avg-pH').textContent = avgFeatures["pH Min"]?.toFixed(2) ?? "-";
    document.getElementById('avg-conductivity').textContent = avgFeatures["Conductivity (µmhos/cm) Min"]?.toFixed(2) ?? "-";
    document.getElementById('avg-bod').textContent = avgFeatures["B.O.D. (mg/l) Min"]?.toFixed(2) ?? "-";
    document.getElementById('avg-nitrate').textContent = avgFeatures["Nitrate-N + Nitrite-N (mg/l) Min"]?.toFixed(2) ?? "-";
}


