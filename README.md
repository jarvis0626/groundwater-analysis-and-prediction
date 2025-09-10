Groundwater Quality Analysis (Flask + ML)

This project is a Flask-based web app that analyzes groundwater quality data using Machine Learning.  
It applies preprocessing (imputation, scaling, PCA) and trains a Decision Tree Regressor to predict and classify groundwater quality for different states across years.  


Features
- Upload & process CSV / Excel groundwater datasets  
- Preprocessing: Missing value imputation, scaling, PCA  
- State-wise groundwater quality classification (Good / Poor)  
- Historical + 3-year future predictions  
- Model evaluation metrics (R², RMSE, MAE)  
- Feature importance visualization  


Tech Stack
- Backend: Flask (Python)  
- ML: scikit-learn (Decision Tree Regressor, PCA)  
- Data Handling: pandas, numpy  
- Frontend: HTML (Jinja2 templates), JavaScript (for API calls)  

Setup & Installation

1. Clone the repository
<pre>
git clone https://github.com/
cd <repo>
</pre>
2. Create and activate a virtual environment
<pre>
  python -m venv venv
</pre>
For Windows
<pre>
venv\Scripts\activate
</pre>
macOS / Linux
<pre>
source venv/bin/activate
</pre>
3. Install dependencies
<pre>
pip install -r requirements.txt
</pre>
5. Run the app
<pre>
python app.py
</pre>
Enjoy!
