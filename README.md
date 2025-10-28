# Fetal Health Classifier

This project tackles the **fetal health classification** problem using **Random Forest**, **XGBoost**, and other ensemble learning methods to achieve better predictive performance.

The notebook focuses on the following key steps:

- **Preprocessing and oversampling** an imbalanced dataset  
- Training ensemble models: **Random Forest**, **AdaBoost**, and **XGBoost**  
- Performing model tuning using **Grid Search** and **Randomized Search**  
- Building a complete and reusable **preprocessing pipeline** with scikit-learn  
- Combining **Grid Search** and the pipeline to automate model evaluation and selection  

## **Dataset**

This project uses the [**Fetal Health Classification dataset**](https://www.kaggle.com/andrewmvd/fetal-health-classification) from Kaggle.  
The dataset supports efforts to reduce **child and maternal mortality**, which are central goals of the **United Nations’ Sustainable Development Agenda**.  
By improving fetal health monitoring, such analyses can help healthcare providers take timely action during pregnancy and childbirth.

**Cardiotocograms (CTGs)** — the data source in this study — are simple, cost-effective tools used to assess fetal well-being.  
CTGs measure fetal heart rate (FHR), uterine contractions, and fetal movements using ultrasound pulses, allowing clinicians to detect potential risks early.

The dataset includes **2,126 records** of features extracted from CTG exams, each labeled by expert obstetricians into three classes:

- **Normal**: 1  
- **Suspect**: 2  
- **Pathological**: 3  

## **Usage**

To run the notebook:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fetal_health_classifier.git
   cd fetal_health_classifier

2. **(Optional) Create and activate a virtual environment**
    ```bash
    python -m venv fhc_env
    source fhc_env/bin/activate      # macOS/Linux  
    fhc_env\Scripts\activate         # Windows  

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

4. **Run the project (Python version)**

	If you want to execute the full pipeline from the command line:
    ```bash
    python -m src.main \
  		--csv data/raw/fetal_health.csv \
  		--model rf \
  		--resample none \
  		--save models/fetal_health_model.joblib \
  		--report results/classification_report.txt
    ```
	You can change the model type (rf, ada, logreg, svc) or try different resampling methods (none, random, smote, adasyn).

5. **(Optional) Run the notebook version**

	If you prefer to explore the results interactively:
    ```bash
    jupyter notebook fetal_health_classifier.ipynb

## **Results and Key Findings**

After experimenting with multiple ensemble models and parameter tuning:
	•	Random Forest and XGBoost achieved the highest overall accuracy and F1-scores.
	•	Oversampling improved the recall for minority classes (Suspect and Pathological).
	•	Integrating preprocessing and model selection in a single pipeline made the workflow clean and reproducible.

Future improvements could include:
	•	Feature importance visualization
	•	Testing additional models like LightGBM or CatBoost
	•	Deploying the trained model through a simple web interface

## **License**

This project is open for educational and research purposes only.
The dataset belongs to the original authors and is available under the Kaggle Dataset License.
You’re free to fork, modify, and learn from this codebase — attribution is appreciated.

