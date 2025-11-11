Machine Learning Model

## Objective

Build and evaluate a machine learning model to predict EV energy consumption (kWh) based on charging behavior (start hour and charging duration). Save the trained model and produce evaluation plots for the Week-3 submission.

---

## Tasks Completed

* Loaded cleaned dataset (`cleaned_ev_dataset.csv`).
* Selected features: `Hour`, `ChargingDuration`.
* Trained Linear Regression and Random Forest Regression models.
* Evaluated models with MAE, RMSE, and R².
* Saved the best model to `models/ev_energy_model.pkl`.
* Generated evaluation plots: Actual vs Predicted, Residuals distribution.

---

## Files Included

* `week3_ml_notebook.ipynb` — Jupyter notebook with full training, evaluation, and plots code.
* `train_model.py` — Python script to train and save the model.
* `ev_model_plots.py` — Script to generate and save evaluation plots.
* `ev_ml_report.pdf` — Week-3 summary (report) ready to download.
* `streamlit_app.py` — Updated Streamlit app with prediction feature.
* `chatbot_faq.py` — Simple project chatbot for FAQs using TF-IDF similarity.

---

## How to Run (Quick)

1. Place `cleaned_ev_dataset.csv` in the project folder.
2. To train the model (notebook or script):

   ```bash
   python train_model.py
   ```
3. To generate plots:

   ```bash
   python ev_model_plots.py
   ```
4. To run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```
5. To run the chatbot (console):

   ```bash
   python chatbot_faq.py
   ```

---

