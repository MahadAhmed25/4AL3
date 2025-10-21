# AI Usage

### 1. Grid Search for Best Hyperparameters
**Prompt:**  
> “Build a GridSearch function to test different SVM hyperparameters (C, gamma, kernel) and find the best setup using TSS as a scoring metric. I already have the X and y data arrays so pass that as a param into this function. ”

**Used for:**  
Creating a grid search to tune SVM parameters and find the best performing configuration.

---

### 2. Confusion Matrix Summary (All 15 Models)
**Prompt:**  
> “i have a results dict with the name of the feature set and a confusion matrix that looks like this: SVM_results = {
            "combo": combo_name,
            "overall_cm": overall_cm,
            **{f"{k}_mean": mean_metrics[k] for k in keys},
            **{f"{k}_std":  std_metrics[k]  for k in keys},
        } Use Seaborn to plot all 15 confusion matrices from my SVM models in one visual grid layout instead of printing them individually.”

**Used for:**  
Generating a single Seaborn figure with all confusion matrices arranged in a grid.

---

### 3. TSS Bar Chart Visualization
**Prompt:**  
> i have a results dict with the name of the feature set and a confusion matrix that looks like this: SVM_results = {
            "combo": combo_name,
            "overall_cm": overall_cm,
            **{f"{k}_mean": mean_metrics[k] for k in keys},
            **{f"{k}_std":  std_metrics[k]  for k in keys},
        } make a function that sort the values by Tss_mean key and plot it in a horizontal bar graph with TSS_std as the error.

**Used for:**  
Visualizing and ranking feature set combinations based on TSS.

for the sake of simplicity I have provided a more simple version of the prompt I used so this file doesnt get too large.
each prompt took multiple prompts to finetune the output to get it to where I want so ill average at about 3 finetuning prompts per prompt above.
In total 9 prompts

CO2 usage: 4.32g * (query) = 4.32(9) = 38.88g
