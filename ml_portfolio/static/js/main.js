document.getElementById('portfolioForm').addEventListener('submit', function() {
    document.getElementById('loadingOverlay').classList.add('active');
    document.getElementById('submitBtn').disabled = true;

    // Cycle through loading messages
    const messages = [
      "Fetching NSE market data...",
      "Engineering 18 features per stock...",
      "KMeans risk clustering...",
      "LightGBM return prediction...",
      "Ledoit-Wolf covariance estimation...",
      "Optimizing 50 candidate portfolios...",
      "Running Monte Carlo simulations...",
      "Building your allocation table..."
    ];
    let i = 0;
    const el = document.getElementById('loadingSteps');
    setInterval(() => {
      el.innerHTML = messages[i % messages.length];
      i++;
    }, 1800);
  });