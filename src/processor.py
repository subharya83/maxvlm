
from google.colab import userdata
APIKEY=userdata.get('APIKEY')
     

import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
import math
     

element_df = pd.read_csv('Elemental_property_data.csv')
     

element_df
     
Symbol	Atomic number	Group	Period	Density	Electronegativity	Ionisation Energy	Atomic radius	UE
0	H	1	1	1	0.000090	2.20	13.598	120	1
1	He	2	18	1	0.000179	0.00	24.587	140	0
2	Li	3	1	2	0.534000	0.98	5.392	182	1
3	Be	4	2	2	1.850000	1.57	9.323	153	0
4	B	5	13	2	2.370000	2.04	8.298	192	1
...	...	...	...	...	...	...	...	...	...
90	Pa	91	3	7	15.370000	1.50	5.890	243	3
91	U	92	3	7	18.950000	1.38	6.194	240	4
92	Np	93	3	7	20.250000	1.36	6.266	221	5
93	Pu	94	3	7	19.840000	1.28	6.060	243	6
94	Am	95	3	7	13.690000	1.30	5.993	244	7
95 rows × 9 columns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

features = element_df[['Atomic number', 'Group', 'Period', 'Density',
       'Electronegativity', 'UE', 'Ionisation Energy', 'Atomic radius']]
scaled_features = StandardScaler().fit_transform(features)

squared_features = np.square(scaled_features)
extended_features = np.hstack((scaled_features, squared_features))

element_df['vector'] = list(extended_features)
     



element_df['vector'][0]
     
array([-1.71391365, -1.25962445, -2.52182563, -1.28035317,  0.73052027,
       -0.63634077,  1.60645454, -2.28696925,  2.9375    ,  1.58665375,
        6.35960449,  1.63930425,  0.53365987,  0.40492958,  2.58069618,
        5.23022834])

element_df = element_df.filter(items=['Symbol','vector'])
element_df
     
Symbol	vector
0	H	[-1.713913650100261, -1.2596244465496966, -2.5...
1	He	[-1.6774474022257875, 1.6674165335621884, -2.5...
2	Li	[-1.6409811543513138, -1.2596244465496966, -1....
3	Be	[-1.6045149064768403, -1.0874455653666446, -1....
4	B	[-1.5680486586023665, 0.8065221276469281, -1.8...
...	...	...
90	Pa	[1.5680486586023665, -0.9152666841835926, 1.42...
91	U	[1.6045149064768403, -0.9152666841835926, 1.42...
92	Np	[1.6409811543513138, -0.9152666841835926, 1.42...
93	Pu	[1.6774474022257875, -0.9152666841835926, 1.42...
94	Am	[1.713913650100261, -0.9152666841835926, 1.427...
95 rows × 2 columns


element_df.rename(columns={'Symbol': 'Element', 'vector': 'Vector'}, inplace=True)
     

compound_df=  pd.read_excel('Compound_property_data.xlsx')
     

compound_df
     
nsites	Chemsys	material_id	ordering	total_magnetization	total_magnetization_normalized_vol	total_magnetization_normalized_atoms	formation_energy_per_atom	energy_above_hull
0	5	Ac-Cr-O	mp-866101	FM	3.000000	0.048890	0.600000	-3.138972	0
1	4	Ac-Au-Eu	mp-1006278	FM	7.413205	0.066629	1.853301	-0.779867	0
2	4	Ac-Eu-Zn	mp-1183124	FM	7.023200	0.062251	1.755800	-0.261696	0
3	5	Ac-Fe-O	mp-861502	FM	4.254276	0.068842	0.850855	-2.771539	0
4	5	Ac-Mn-O	mp-864911	FM	3.999993	0.064266	0.799999	-2.973630	0
...	...	...	...	...	...	...	...	...	...
5949	18	Sb-U-Zr	mp-12889	FM	15.857187	0.033430	0.880955	-0.746757	0
5950	4	N-U-Zr	mp-1215248	FM	2.095059	0.039252	0.523765	-1.840461	0
5951	4	S-U-Zr	mp-1215176	FM	2.742706	0.035114	0.685677	-1.662938	0
5952	12	Fe-V-Zr	mp-1215261	FiM	2.937033	0.015563	0.244753	-0.187117	0
5953	6	Zn-Zr	mp-1401	FM	2.207073	0.022170	0.367846	-0.368674	0
5954 rows × 9 columns


categories_to_keep = ['FM','FiM']
compound_df = compound_df[compound_df['ordering'].isin(categories_to_keep)]
     

counts = compound_df['ordering'].value_counts()
counts
     
count
ordering	
FM	4483
FiM	1258

dtype: int64

unique_categories = compound_df['ordering'].unique()
print(unique_categories)
     
['FM' 'FiM']
Feature engineering

outer_product_results = []

def calculate_outer_products_with_elemental_stats(structure, cutoff_radius, element_df, supercell_size):
    original_structure = structure.copy()
    original_indices = list(range(len(original_structure)))
    supercell = structure * supercell_size

    tolerance = 1e-6

    a1, a2, a3 = structure.lattice.matrix

    original_cart_coords = [site.coords for site in structure]

    original_indices_in_supercell = []

    for i, site in enumerate(supercell):
        for orig in original_cart_coords:
            if np.linalg.norm(site.coords - orig) < tolerance:
                original_indices_in_supercell.append(i)
                break


    elements = supercell.species
    outer_product_matrices = []
    distances = []
    en_differences = []
    en_sq=[]

    for i in original_indices_in_supercell:
        for j in range(len(supercell)):
            distance = supercell.get_distance(i, j)

            if distance <= cutoff_radius and distance > 0:
                row_vector_i = element_df.loc[element_df['Element'] == str(elements[i]), 'Vector'].values[0]
                row_vector_j = element_df.loc[element_df['Element'] == str(elements[j]), 'Vector'].values[0]

                outer_product = np.outer(row_vector_i, row_vector_j)/(distance*distance)

                outer_product_matrices.append(outer_product)

                distances.append(distance)

                en_i = row_vector_i[4]
                en_j = row_vector_j[4]
                en_diff1 = abs(en_i - en_j)
                en_diff2 = abs(en_i*en_i - en_j*en_j)

                en_differences.append(en_diff1)
                en_sq.append(en_diff2)

    outer_product_matrices = np.array(outer_product_matrices)

    mean_matrix = np.mean(outer_product_matrices, axis=0)
    std_matrix = np.std(outer_product_matrices, axis=0)

    dist_mean = np.mean(distances)
    dist_std = np.std(distances)

    en_diff_mean = np.mean(en_differences)
    en_diff_std = np.std(en_differences)
    en_sq_mean = np.mean(en_sq)
    en_sq_std = np.std(en_sq)

    elemental_vectors = []

    flattened_mean = mean_matrix.flatten()
    flattened_std = std_matrix.flatten()

    feature_vector = np.concatenate([flattened_mean, flattened_std, [dist_mean, dist_std], [en_diff_mean, en_diff_std], [en_sq_mean, en_sq_std]])

    return feature_vector

cutoff_radius = 5
supercell_size = (4, 4, 4)

for _, row in compound_df.iterrows():
    material_id = row['material_id']

    try:
        with MPRester(APIKEY) as m:
            structure = m.get_structure_by_material_id(material_id)
            print(f"Processing material: {material_id}")

        feature_vector = calculate_outer_products_with_elemental_stats(structure, cutoff_radius, element_df, supercell_size)

        outer_product_results.append({'material_id': material_id, 'features': feature_vector})

    except Exception as e:
        print(f"Error processing material {material_id}: {e}")
        continue

df_features = pd.DataFrame(outer_product_results)


     material_id                                           features
0      mp-866101  [0.034564520781301906, -0.021964951645489516, ...
1     mp-1006278  [0.09116477267570224, -0.020771136032048304, 0...
2     mp-1183124  [-0.016525383245708087, 0.012654252961144582, ...
3      mp-861502  [0.032457623360230295, -0.025593760616601038, ...
4      mp-864911  [0.033268739456372334, -0.023624416551478806, ...
...          ...                                                ...
5735    mp-12889  [0.019908583060276156, 0.017317300818035418, 0...
5736  mp-1215248  [0.008572120209416702, -0.00042085987774751333...
5737  mp-1215176  [-0.0044305417367528944, 0.00773183437402287, ...
5738  mp-1215261  [0.035089160602986495, 0.02517746244475388, 0....
5739     mp-1401  [0.0215573699545378, -0.005654499608002252, 0....

[5740 rows x 2 columns]

compound_df
     
nsites	Chemsys	material_id	ordering	total_magnetization	total_magnetization_normalized_vol	total_magnetization_normalized_atoms	formation_energy_per_atom	energy_above_hull
0	5	Ac-Cr-O	mp-866101	FM	3.000000	0.048890	0.600000	-3.138972	0
1	4	Ac-Au-Eu	mp-1006278	FM	7.413205	0.066629	1.853301	-0.779867	0
2	4	Ac-Eu-Zn	mp-1183124	FM	7.023200	0.062251	1.755800	-0.261696	0
3	5	Ac-Fe-O	mp-861502	FM	4.254276	0.068842	0.850855	-2.771539	0
4	5	Ac-Mn-O	mp-864911	FM	3.999993	0.064266	0.799999	-2.973630	0
...	...	...	...	...	...	...	...	...	...
5949	18	Sb-U-Zr	mp-12889	FM	15.857187	0.033430	0.880955	-0.746757	0
5950	4	N-U-Zr	mp-1215248	FM	2.095059	0.039252	0.523765	-1.840461	0
5951	4	S-U-Zr	mp-1215176	FM	2.742706	0.035114	0.685677	-1.662938	0
5952	12	Fe-V-Zr	mp-1215261	FiM	2.937033	0.015563	0.244753	-0.187117	0
5953	6	Zn-Zr	mp-1401	FM	2.207073	0.022170	0.367846	-0.368674	0
5741 rows × 9 columns

Magnetic ordering prediction

result = df_features.merge(compound_df[['material_id', 'ordering']], on='material_id', how='left')
result = result.rename(columns={'features': 'outer_product'})
print(result)
     
     material_id                                      outer_product ordering
0      mp-866101  [0.034564520781301906, -0.021964951645489516, ...       FM
1     mp-1006278  [0.09116477267570224, -0.020771136032048304, 0...       FM
2     mp-1183124  [-0.016525383245708087, 0.012654252961144582, ...       FM
3      mp-861502  [0.032457623360230295, -0.025593760616601038, ...       FM
4      mp-864911  [0.033268739456372334, -0.023624416551478806, ...       FM
...          ...                                                ...      ...
5735    mp-12889  [0.019908583060276156, 0.017317300818035418, 0...       FM
5736  mp-1215248  [0.008572120209416702, -0.00042085987774751333...       FM
5737  mp-1215176  [-0.0044305417367528944, 0.00773183437402287, ...       FM
5738  mp-1215261  [0.035089160602986495, 0.02517746244475388, 0....      FiM
5739     mp-1401  [0.0215573699545378, -0.005654499608002252, 0....       FM

[5740 rows x 3 columns]

result['ordering'] = encoder.fit_transform(result['ordering'])
     

print("Class mappings:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))
     
Class mappings: {'FM': 0, 'FiM': 1}



from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['ordering']], axis=1)

X = df_features_expanded.drop('ordering', axis=1)
y = df_features_expanded['ordering']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class_weight = {0: 1, 1: 3}
model = LGBMClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=10,
    class_weight=class_weight,
    min_child_samples=10,
    subsample=0.5,
    reg_alpha=0.1,
    reg_lambda=0.3,
    colsample_bytree=0.3,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Testing Classification Report:")
print(classification_report(y_test, y_test_pred))
     
[LightGBM] [Info] Number of positive: 1148, number of negative: 4018
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.027614 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 5166, number of used features: 518
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.461538 -> initscore=-0.154151
[LightGBM] [Info] Start training from score -0.154151
Training Accuracy: 0.854045683313976
Training Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.87      0.90      4018
           1       0.63      0.81      0.71      1148

    accuracy                           0.85      5166
   macro avg       0.79      0.84      0.81      5166
weighted avg       0.87      0.85      0.86      5166

Testing Accuracy: 0.8414634146341463
Testing Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.86      0.90       465
           1       0.56      0.74      0.64       109

    accuracy                           0.84       574
   macro avg       0.75      0.80      0.77       574
weighted avg       0.86      0.84      0.85       574


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print(f"Training ROC-AUC: {train_roc_auc}")
print(f"Testing ROC-AUC: {test_roc_auc}")

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color='blue', label=f'Training ROC curve (AUC = {train_roc_auc:.2f})')
plt.plot(fpr_test, tpr_test, color='red', label=f'Testing ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
     
Training ROC-AUC: 0.9269958748350194
Testing ROC-AUC: 0.8833579954621682

70:30


from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['ordering']], axis=1)

X = df_features_expanded.drop('ordering', axis=1)
y = df_features_expanded['ordering']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class_weight = {0: 2, 1: 7}
model = LGBMClassifier(
    n_estimators=25,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=9,
    class_weight=class_weight,
    min_child_samples=12,
    subsample=0.4,
    reg_alpha=0.1,
    reg_lambda=0.3,
    colsample_bytree=0.3,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Testing Classification Report:")
print(classification_report(y_test, y_test_pred))
     
[LightGBM] [Info] Number of positive: 881, number of negative: 3137
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.026896 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 4018, number of used features: 518
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.495700 -> initscore=-0.017202
[LightGBM] [Info] Start training from score -0.017202
Training Accuracy: 0.8163265306122449
Training Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.82      0.87      3137
           1       0.56      0.80      0.66       881

    accuracy                           0.82      4018
   macro avg       0.75      0.81      0.77      4018
weighted avg       0.85      0.82      0.83      4018

Testing Accuracy: 0.8019744483159117
Testing Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.82      0.87      1346
           1       0.53      0.74      0.62       376

    accuracy                           0.80      1722
   macro avg       0.73      0.78      0.74      1722
weighted avg       0.83      0.80      0.81      1722


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print(f"Training ROC-AUC: {train_roc_auc}")
print(f"Testing ROC-AUC: {test_roc_auc}")

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color='blue', label=f'Training ROC curve (AUC = {train_roc_auc:.2f})')
plt.plot(fpr_test, tpr_test, color='red', label=f'Testing ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
     
Training ROC-AUC: 0.8991436108951162
Testing ROC-AUC: 0.846440201068572

Magnetic moment prediction

result = df_features.merge(compound_df[['material_id', 'total_magnetization_normalized_atoms']], on='material_id', how='left')
result = result.rename(columns={'features': 'outer_product'})
print(result)
     
     material_id                                      outer_product  \
0      mp-866101  [0.034564520781301906, -0.021964951645489516, ...   
1     mp-1006278  [0.09116477267570224, -0.020771136032048304, 0...   
2     mp-1183124  [-0.016525383245708087, 0.012654252961144582, ...   
3      mp-861502  [0.032457623360230295, -0.025593760616601038, ...   
4      mp-864911  [0.033268739456372334, -0.023624416551478806, ...   
...          ...                                                ...   
5735    mp-12889  [0.019908583060276156, 0.017317300818035418, 0...   
5736  mp-1215248  [0.008572120209416702, -0.00042085987774751333...   
5737  mp-1215176  [-0.0044305417367528944, 0.00773183437402287, ...   
5738  mp-1215261  [0.035089160602986495, 0.02517746244475388, 0....   
5739     mp-1401  [0.0215573699545378, -0.005654499608002252, 0....   

      total_magnetization_normalized_atoms  
0                                 0.600000  
1                                 1.853301  
2                                 1.755800  
3                                 0.850855  
4                                 0.799999  
...                                    ...  
5735                              0.880955  
5736                              0.523765  
5737                              0.685677  
5738                              0.244753  
5739                              0.367846  

[5740 rows x 3 columns]
Linear regression - Magnetic moment


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['total_magnetization_normalized_atoms']], axis=1)

X = df_features_expanded.drop('total_magnetization_normalized_atoms', axis=1)
y = df_features_expanded['total_magnetization_normalized_atoms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("TRAINING SET PERFORMANCE:")
print("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, y_train_pred))
print("R² Score (Train):", r2_score(y_train, y_train_pred))

print("\nTEST SET PERFORMANCE:")
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, y_test_pred))
print("R² Score (Test):", r2_score(y_test, y_test_pred))
     
TRAINING SET PERFORMANCE:
Mean Squared Error (Train): 0.09967919271225134
Mean Absolute Error (Train): 0.21849328358322456
R² Score (Train): 0.8529320793516981

TEST SET PERFORMANCE:
Mean Squared Error (Test): 0.13197689505542118
Mean Absolute Error (Test): 0.25810269971641736
R² Score (Test): 0.7770885108303968
LightGBM - Magnetic Moment

90:10


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['total_magnetization_normalized_atoms']], axis=1)

X = df_features_expanded.drop('total_magnetization_normalized_atoms', axis=1)
y = df_features_expanded['total_magnetization_normalized_atoms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=8,
    num_leaves=18,
    subsample=0.5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    colsample_bytree=0.25,
    min_child_samples=15,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def correlation_coefficient(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

print("TRAINING SET PERFORMANCE:")
print("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (Train):", rmse(y_train, y_train_pred))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, y_train_pred))
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Correlation Coefficient (Train):", correlation_coefficient(y_train, y_train_pred))

print("\nTEST SET PERFORMANCE:")
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (Test):", rmse(y_test, y_test_pred))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, y_test_pred))
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Correlation Coefficient (Test):", correlation_coefficient(y_test, y_test_pred))
     
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.030370 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 5166, number of used features: 518
[LightGBM] [Info] Start training from score 0.732051
TRAINING SET PERFORMANCE:
Mean Squared Error (Train): 0.028853132055478244
Root Mean Squared Error (Train): 0.16986209717143563
Mean Absolute Error (Train): 0.12421222275474542
R² Score (Train): 0.957429730115897
Correlation Coefficient (Train): 0.9791646103455008

TEST SET PERFORMANCE:
Mean Squared Error (Test): 0.07610359110686997
Root Mean Squared Error (Test): 0.27586879328200564
Mean Absolute Error (Test): 0.19027457124266284
R² Score (Test): 0.8714595852731413
Correlation Coefficient (Test): 0.9338949275286276

import matplotlib.pyplot as plt

max_limit = max(max(y_test), max(y_test_pred))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, s=25, edgecolor='none')
plt.plot([0, max_limit], [0, max_limit], color='red', linestyle='--', lw=2)

plt.xlim(0, max_limit)
plt.ylim(0, max_limit)

plt.xlabel("Actual Magnetic Moment per Atom [μB/Atom]")
plt.ylabel("Predicted Magnetic Moment per Atom [μB/Atom]")
plt.grid(False)
plt.show()
     

70:30


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['total_magnetization_normalized_atoms']], axis=1)

X = df_features_expanded.drop('total_magnetization_normalized_atoms', axis=1)
y = df_features_expanded['total_magnetization_normalized_atoms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=75,
    learning_rate=0.08,
    max_depth=8,
    num_leaves=12,
    subsample=0.30,
    reg_alpha=0.5,
    reg_lambda=0.5,
    colsample_bytree=0.3,
    min_child_samples=15,
    random_state=42
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

print("TRAINING SET PERFORMANCE:")
print("Root Mean Squared Error (Train):", rmse(y_train, y_train_pred))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, y_train_pred))
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Pearson Correlation Coefficient (Train):", pearson_corr(y_train, y_train_pred))

print("\nTEST SET PERFORMANCE:")
print("Root Mean Squared Error (Test):", rmse(y_test, y_test_pred))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, y_test_pred))
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Pearson Correlation Coefficient (Test):", pearson_corr(y_test, y_test_pred))
     
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.022846 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 4018, number of used features: 518
[LightGBM] [Info] Start training from score 0.735971
TRAINING SET PERFORMANCE:
Root Mean Squared Error (Train): 0.27390722356149827
Mean Absolute Error (Train): 0.20101062608231163
R² Score (Train): 0.8897118097601765
Pearson Correlation Coefficient (Train): 0.9450612894354673

TEST SET PERFORMANCE:
Root Mean Squared Error (Test): 0.3586764219173447
Mean Absolute Error (Test): 0.24417527277318185
R² Score (Test): 0.8000204448540755
Pearson Correlation Coefficient (Test): 0.8947761927960604

import matplotlib.pyplot as plt

max_limit = max(max(y_test), max(y_test_pred))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, s=25, edgecolor='none')
plt.plot([0, max_limit], [0, max_limit], color='red', linestyle='--', lw=2)

plt.xlim(0, max_limit)
plt.ylim(0, max_limit)

plt.xlabel("Actual Magnetic Moment per Atom [μB/Atom]")
plt.ylabel("Predicted Magnetic Moment per Atom [μB/Atom]")
plt.grid(False)
plt.show()
     

Formation energy prediction

result = df_features.merge(compound_df[['material_id', 'formation_energy_per_atom']], on='material_id', how='left')
result = result.rename(columns={'features': 'outer_product'})
print(result)
     
     material_id                                      outer_product  \
0      mp-866101  [0.034564520781301906, -0.021964951645489516, ...   
1     mp-1006278  [0.09116477267570224, -0.020771136032048304, 0...   
2     mp-1183124  [-0.016525383245708087, 0.012654252961144582, ...   
3      mp-861502  [0.032457623360230295, -0.025593760616601038, ...   
4      mp-864911  [0.033268739456372334, -0.023624416551478806, ...   
...          ...                                                ...   
5735    mp-12889  [0.019908583060276156, 0.017317300818035418, 0...   
5736  mp-1215248  [0.008572120209416702, -0.00042085987774751333...   
5737  mp-1215176  [-0.0044305417367528944, 0.00773183437402287, ...   
5738  mp-1215261  [0.035089160602986495, 0.02517746244475388, 0....   
5739     mp-1401  [0.0215573699545378, -0.005654499608002252, 0....   

      formation_energy_per_atom  
0                     -3.138972  
1                     -0.779867  
2                     -0.261696  
3                     -2.771539  
4                     -2.973630  
...                         ...  
5735                  -0.746757  
5736                  -1.840461  
5737                  -1.662938  
5738                  -0.187117  
5739                  -0.368674  

[5740 rows x 3 columns]
90:10


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['formation_energy_per_atom']], axis=1)

X = df_features_expanded.drop('formation_energy_per_atom', axis=1)
y = df_features_expanded['formation_energy_per_atom']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=30,
    subsample=0.75,
    reg_alpha=1.0,
    reg_lambda=1.0,
    colsample_bytree=0.6,
    min_child_samples=60,
    random_state=42
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def correlation_coefficient(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

print("TRAINING SET PERFORMANCE:")
print("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (Train):", rmse(y_train, y_train_pred))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, y_train_pred))
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Correlation Coefficient (Train):", correlation_coefficient(y_train, y_train_pred))

print("\nTEST SET PERFORMANCE:")
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (Test):", rmse(y_test, y_test_pred))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, y_test_pred))
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Correlation Coefficient (Test):", correlation_coefficient(y_test, y_test_pred))
     
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.036090 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 5166, number of used features: 518
[LightGBM] [Info] Start training from score -1.136775
TRAINING SET PERFORMANCE:
Mean Squared Error (Train): 0.019536908845699308
Root Mean Squared Error (Train): 0.13977449282934032
Mean Absolute Error (Train): 0.1016005580103043
R² Score (Train): 0.9786015271583188
Correlation Coefficient (Train): 0.9895150701169265

TEST SET PERFORMANCE:
Mean Squared Error (Test): 0.0390085928981594
Root Mean Squared Error (Test): 0.19750593129868127
Mean Absolute Error (Test): 0.13817654539684718
R² Score (Test): 0.9569385092930813
Correlation Coefficient (Test): 0.9784235120180782

min_val = min(min(y_test), min(y_test_pred))
min_tick = int(np.floor(min_val))

ticks = np.arange(min_tick, 1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, s=25, edgecolor='none')
plt.plot([min_tick, 0], [min_tick, 0], color='red', linestyle='--', lw=2)

plt.xlim(min_tick, 0)
plt.ylim(min_tick, 0)

plt.xticks(ticks)
plt.yticks(ticks)

plt.xlabel("Actual Formation Energy per Atom [eV/Atom]")
plt.ylabel("Predicted Formation Energy per Atom [eV/Atom]")
plt.grid(False)
plt.show()
     

70:30


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

outer_product_df = pd.DataFrame(result['outer_product'].tolist(), columns=[f'feature_{i+1}' for i in range(518)])
df_features_expanded = pd.concat([outer_product_df, result['formation_energy_per_atom']], axis=1)

X = df_features_expanded.drop('formation_energy_per_atom', axis=1)
y = df_features_expanded['formation_energy_per_atom']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=30,
    subsample=0.75,
    reg_alpha=1.0,
    reg_lambda=1.0,
    colsample_bytree=0.6,
    min_child_samples=60,
    random_state=42
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def correlation_coefficient(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

print("TRAINING SET PERFORMANCE:")
print("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (Train):", rmse(y_train, y_train_pred))
print("Mean Absolute Error (Train):", mean_absolute_error(y_train, y_train_pred))
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Correlation Coefficient (Train):", correlation_coefficient(y_train, y_train_pred))

print("\nTEST SET PERFORMANCE:")
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (Test):", rmse(y_test, y_test_pred))
print("Mean Absolute Error (Test):", mean_absolute_error(y_test, y_test_pred))
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Correlation Coefficient (Test):", correlation_coefficient(y_test, y_test_pred))
     
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.028605 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 132090
[LightGBM] [Info] Number of data points in the train set: 4018, number of used features: 518
[LightGBM] [Info] Start training from score -1.131746
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
TRAINING SET PERFORMANCE:
Mean Squared Error (Train): 0.01938327715935086
Root Mean Squared Error (Train): 0.13922383833004626
Mean Absolute Error (Train): 0.0993143573840436
R² Score (Train): 0.9785217358786304
Correlation Coefficient (Train): 0.9895254133363517

TEST SET PERFORMANCE:
Mean Squared Error (Test): 0.038282523730034115
Root Mean Squared Error (Test): 0.1956592030292317
Mean Absolute Error (Test): 0.1378615433648498
R² Score (Test): 0.9590563885921242
Correlation Coefficient (Test): 0.9798591674069442

min_val = min(min(y_test), min(y_test_pred))
min_tick = int(np.floor(min_val))

ticks = np.arange(min_tick, 1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, s=25, edgecolor='none')
plt.plot([min_tick, 0], [min_tick, 0], color='red', linestyle='--', lw=2)

plt.xlim(min_tick, 0)
plt.ylim(min_tick, 0)

plt.xticks(ticks)
plt.yticks(ticks)

plt.xlabel("Actual Formation Energy per Atom [eV/Atom]")
plt.ylabel("Predicted Formation Energy per Atom [eV/Atom]")
plt.grid(False)
plt.show()