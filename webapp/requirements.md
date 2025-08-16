# Your task now is to create an app to help users predict the risk of dialysis or all cause mortality
- you should use the models in /mnt/dump/yard/projects/tarot2/foundation_models
- all the models config is in /mnt/dump/yard/projects/tarot2/results/final_deploy/model_config
- all the documentation in /mnt/dump/yard/projects/tarot2/docs can be used as reference

# potential users
- healthcare professionals
- patients

# Input data processing
# Necessary
The user should input:
1. his/her age OR date of birth
2. gender
3. latest creatinine and the date of investigation
4. latest hemoglobin and the date of investigation
5. latest phosphate and the date of investigation
6. latest bicarbonate and the date of observation
7. latest urine albumin creatinine ratio (uacr) and the date of investigation OR 
8. latest urine protein creatinine ratio (upcr) and the date of investigation
* please note that for item 3 to 8 you may need to design function to allow user to choose the unit for input e.g. someone will input creatinine in mg/dL but some will use umol/L *
* You need you pay attention to what unit used by the models in this repo *
* You can get the detail of units used in this repo in /mnt/dump/yard/projects/tarot2/src/default_data_validation_rules.yml *
* You need to provide conversion if people using different units to input (e.g. a dropdown list to let user select the input they use) *
* for item 7 and 8, item 7 has higher priority: if one input both item 7 AND item 8, use item 7*
* if item 8 is used, please read the codebase and do the conversion to convert upcr to uacr. *

# Non-necessary (can ask the patient to inpute data as long as he/she can remember, but can be left blank)
A. History of hypertension (yes/no) with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
B. History of diabetes (yes/no) with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
C. History of myocardial infarction with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
D. History of congestive heart failure with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
E. History of peripheral vascular disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
F. History of cerebrovascular disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
G. History of dementia with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
H. History of chronic pulmonary disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
I. History of rheumatic disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
J. History of peptic ulcer disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
K. History of mild liver disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
L. History of renal disease (mild to moderate) with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
M. History of diabetes complications with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
N. History of hemiplegia/paraplegia with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
O. History of any malignancy with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
P. History of severe liver disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
Q. Histroy of severe renal disease with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
R. History of hiv with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
S. History of metastatic cancer with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
T. History of aids with date of diagnosis (if user cannot remember the date then the date of user using this is set to date of diagnosis)
* item B to T should be used to calculate Charlson Comorbidity score, you can search the codebase to see how to calculate *

# processing of the user input
1. Collect all the user input data
2. sort all the input according to the date of diagnosis/investigation
3. do the unit conversion to match /mnt/dump/yard/projects/tarot2/docs/unit_conversion_reference.md
4. Calculate the age of user when the diagnosis/observation
5. Now you should be able to arrange the avaliable input in the format of the following dataframe (you can leave the empty cell as you be at this moment):

| Date | Age_at_obs | Gender | creatinine | hemoglobin | phosphate | bicarbonate | uacr | upcr | ht | myocardial_infarction | congestive_heart_failure | peripheral_vascular_disease | cerebrovascular_disease | dementia | chronic_pulmonary_disease | rheumatic_disease | peptic_ulcer_disease | mild_liver_disease | diabetes_wo_complication | diabetes_w_complication | hemiplegia_paraplegia | any_malignancy | liver_severe | metastatic_cancer | hiv | aids |
|------|------------|--------|------------|------------|-----------|-------------|------|------|----|-----------------------|--------------------------|-----------------------------|-------------------------|----------|--------------------------|-------------------|----------------------|-------------------|-------------------------|------------------------|----------------------|----------------|--------------|-------------------|-----|------|
| Date 1 (earliest) | ...    | ...    | ...        | ...        | ...       | ...         | ...  | ...  | ...| ...                   | ...                      | ...                         | ...                     | ...      | ...                      | ...               | ...                  | ...               | ...                     | ...                    | ...                  | ...            | ...          | ...               | ... | ...  |
| Date 2            | ...    | ...    | ...        | ...        | ...       | ...         | ...  | ...  | ...| ...                   | ...                      | ...                         | ...                     | ...      | ...                      | ...               | ...                  | ...               | ...                     | ...                    | ...                  | ...            | ...          | ...               | ... | ...  |
| Date 3            | ...    | ...    | ...        | ...        | ...       | ...         | ...  | ...  | ...| ...                   | ...                      | ...                         | ...                     | ...      | ...                      | ...               | ...                  | ...               | ...                     | ...                    | ...                  | ...            | ...          | ...               | ... | ...  |
| Date 4            | ...    | ...    | ...        | ...        | ...       | ...         | ...  | ...  | ...| ...                   | ...                      | ...                         | ...                     | ...      | ...                      | ...               | ...                  | ...               | ...                     | ...                    | ...                  | ...            | ...          | ...               | ... | ...  |
| Date 5            | ...    | ...    | ...        | ...        | ...       | ...         | ...  | ...  | ...| ...                   | ...                      | ...                         | ...                     | ...      | ...                      | ...               | ...                  | ...               | ...                     | ...                    | ...                  | ...            | ...          | ...               | ... | ...  |

6. base on item B to T in the columns, calculate the Charlson Comorbidity index, add the column 'cci_score_total' as new column
6. now go for data imputation and preprocessing, you can read /mnt/dump/yard/projects/tarot2/steps/impute_data.py and /mnt/dump/yard/projects/tarot2/steps/preprocess_data.py as a reference, the out of box preprocessor is in /mnt/dump/yard/projects/tarot2/foundation_models/ckd_preprocessor.pkl
7. Then slice out the last 10 rows, which should contain the latest 10 rows of dates, and the 'features' stated in /mnt/dump/yard/projects/tarot2/src/default_master_df_mapping.yml 'features' key.
8. after the steps above, you should have 10 rows, and columns of features according to /mnt/dump/yard/projects/tarot2/src/default_master_df_mapping.yml (by default in this repo we have 11 columns)

# Get teh CIF predictions
1. input the feature dataframe/array above to the models
2. get the CIF array
3. ensemble the CIF array in shape of (2, 5, 1) --> 2 endpoints, 5 time horizons, 1 patient
4. Also get the SHAP value for each feature input to the models, each feature should have its SHAP value to Event 1 and Event 2 respectively

# Presentation
## ways of usage
1. a webpage --> mainly aim at patients and laymen
2. API --> mainly aim at healthcare professionals and institutes
3. MCP server --> mainly aim at healthcare professionals and institutes
4. mobile app <-- don't proceed this yet, we will start developthis if other parts works fine

## Webpage
The webpage should be interactive
After received the input above, It should present the data as plots:
1. Risk prediction plots
- In a line plot format, X-axis is time horizon (1 - 5 years), y-axis is risk in percentage, Risk of Event 1 (dialysis) and event 2 (all-cause mortality) in the same plot
2. SHAP plot
- In another plot shoul be a SHAP plot to show each feature's impact on Event 1 and Event 2

## For API and MCP server, just return the prediction in json format

## Create all documentation users need to use the webpage, folk the git or deploy in their own server

## Please communicate and ask me if there is any point need clarification, 

## Warning: DO NOT LOG ANY DATA INPUT
- As we are not able to get and fulfill all the data storage and utilization regulations in all countries, what we can do is to avoid logging any data to protect the privacy of our users