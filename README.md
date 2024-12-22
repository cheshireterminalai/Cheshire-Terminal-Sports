# Cheshire-Terminal-Sports
The Official Git Of The Cheshire Terminal Sports Client
How to Run
Install the required Python libraries if you don’t have them yet:
bash
Copy code
pip install pandas seaborn scikit-learn matplotlib
Check that you have your Kaggle data.csv file (Kobe dataset) in ./kobe-bryant-shot-selection/ or set a custom path in the function call.
(Optional) Place the fullcourt.png file in the same folder if you want to enable EDA shots plotting (set SHOW_PLOTS = True in the script).
Run the script:

bash
Copy code
python cheshire_kobe_bryant_model.py
It will print out:
Baseline accuracy (always predicting miss).
Test accuracy from the hold-out set after hyperparameter tuning.
A quick summary of top feature importances.
You can now integrate cheshire_kobe_bryant_model or the final trained model (rfc_final) into your Solana Sports Agent pipeline, or wherever else you’d like to use it. Have fun exploring new ideas (e.g., different binning, additional features, alternative models)!

Key Takeaways
Baseline accuracy is around 55% (since Kobe missed ~55% of his shots).
With our Random Forest, we get an accuracy in the high 60s on hold-out data.
Feature engineering (distance, angle, time bins, shot type, etc.) significantly improves accuracy.
Hyperparameter tuning via GridSearchCV helps find the best random forest configuration.
The approach is easily extensible—feel free to add or refine features to push accuracy further.
That’s it! You’ve got a reproducible, end-to-end script for Kobe’s shot prediction, including data cleaning, EDA, feature engineering, model building, and final feature importance inspection.

Good luck, and enjoy your Cheshire Terminal journey!
