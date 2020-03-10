#%% import packages
import pandas as pd
import category_encoders as ce
import numpy as np
from pandas_profiling import ProfileReport
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from biokit import corrplot
import scipy
import scipy.cluster.hierarchy as sch
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import patsy

#%% Reading data
og_training = pd.read_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/train/train.csv")
breed_label = pd.read_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/breed_labels.csv")
color_label = pd.read_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/color_labels.csv")
state_label = pd.read_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/state_labels.csv")

#%% merge breed, color, and state labels to the dataset
breed1_merge = pd.merge(og_training, breed_label[["BreedID", "BreedName"]], how = "left", left_on = "Breed1", right_on = "BreedID")
breed1_merge.drop("BreedID", axis = 1, inplace = True)
breed1_merge.BreedName = breed1_merge.BreedName.fillna("Missing")

breed2_merge = pd.merge(breed1_merge, breed_label[["BreedID", "BreedName"]], how = "left", left_on = "Breed2", right_on = "BreedID", suffixes = ("_1", "_2"))
breed2_merge.drop("BreedID", axis = 1, inplace = True)
breed2_merge.BreedName_2 = breed2_merge.BreedName_2.fillna("Missing")

color_merge1 = pd.merge(breed2_merge, color_label, how = "left", left_on = "Color1", right_on = "ColorID")
color_merge1.drop("ColorID", axis = 1, inplace = True)

color_merge2 = pd.merge(color_merge1, color_label, how = "left", left_on = "Color2", right_on = "ColorID", suffixes = ("_1", "_2"))
color_merge2.drop("ColorID", axis = 1, inplace = True)
color_merge2.ColorName_2 = color_merge2.ColorName_2.fillna("Missing")

color_merge3 = pd.merge(color_merge2, color_label, how = "left", left_on = "Color3", right_on = "ColorID")
color_merge3.drop("ColorID", axis = 1, inplace = True)
color_merge3 = color_merge3.rename(columns = {"ColorName": "ColorName_3"})
color_merge3.ColorName_3 = color_merge3.ColorName_3.fillna("Missing")

state_merge = pd.merge(color_merge3, state_label, how = "left", left_on = "State", right_on = "StateID")
state_merge.drop("StateID", axis = 1, inplace = True)

drop_columns = ["Breed1", "Breed2", "Color1", "Color2", "Color3", "State", "RescuerID",
                "Description", "PetID", "Name"]
state_merge.drop(drop_columns, axis = 1, inplace = True)

# state_merge.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/tableau.csv")
gender_merge = state_merge.copy()
gender_merge["Gender"] = gender_merge.apply(lambda x: "Male" if x["Gender"] == 1 else "Female", axis = 1)

type_merge = gender_merge.copy()
type_merge["Type"] = type_merge.apply(lambda x: "Dog" if x["Type"] == 1 else "Cat", axis = 1)

size_merge = type_merge.copy()
size_merge.replace({"MaturitySize": { 1: "Small", 2: "Medium", 3: "Large", 4: "ExtraLarge", 0: "NotSpecified"}}, inplace = True)

fur_merge = size_merge.copy()
fur_merge.replace({"FurLength": {1: "Short", 2: "Medium", 3: "Long", 0: "NotSpecified"}}, inplace = True)

vac_merge = fur_merge.copy()
vac_merge.replace({"Vaccinated": {1: "Vaccinated", 2: "Not Vaccinated", 3: "NotSure Vaccinated"}}, inplace = True)

deworm_merge = vac_merge.copy()
deworm_merge.replace({"Dewormed": {1: "Dewormed", 2: "Not Dewormed", 3: "NotSure Dewormed"}}, inplace = True)

ster_merge = deworm_merge.copy()
ster_merge.replace({"Sterilized": {1: "Neutered", 2: "Not Neutered", 3: "NotSure Neutered"}}, inplace = True)

health_merge = ster_merge.copy()
health_merge.replace({"Health": {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury", 0: "NotSpecified"}}, inplace = True)

health_merge["Adopted"] = health_merge.apply(lambda x: "NoAfter100" if x["AdoptionSpeed"] == 4 else "Yes", axis = 1)

speed_merge = health_merge.copy()
speed_merge.replace({"AdoptionSpeed": {0: "SameDay", 1: "1stWeek", 2: "1stMonth", 3: "2nd3rdMonth", 4: "NoAfter100"}}, inplace = True)

# remove the rows that has unknown vaccination, nuetering, and deworming
data = speed_merge.copy()
data2 = data[(data["Vaccinated"] != "NotSure Vaccinated") & (data["Sterilized"] != "NotSure Neutered") & (data["Dewormed"] != "NotSure Dewormed")]

# add cfa certified breed information
cfa_breeds = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair',
              'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla',
              'Chartreux', 'Colorpoint Shorthair', 'Cornish Rex', 'Devon Rex', 'Egyptian Mau', 'European Burmese',
              'Exotic', 'Havana Brown', 'Japanese Bobtail', 'Korat', 'LaPerm', 'Maine Coon', 'Manx',
              'Norwegian Forest Cat', 'Ocicat', 'Oriental', 'Persian', 'Ragamuffin', 'Ragdoll', 'Russian Blue',
              'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Singapura', 'Somali', 'Tonkinese',
              'Turkish Angora', 'Turkish Van']

cfa_breeds = [i.lower() for i in cfa_breeds]
cfa_breeds = '|'.join(cfa_breeds)


data2["BreedName_1"] = data2["BreedName_1"].str.lower()
data2["BreedName_2"] = data2["BreedName_2"].str.lower()
data2["cfa_breeds"] = np.where((data2["BreedName_1"].str.contains(cfa_breeds)) | (data2["BreedName_2"].str.contains(cfa_breeds)), True, False)

# data2.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/malaysia_animal_data2.csv")

#%% class for some data manipulation
class Cleaning:

    def __init__(self, path, file_path, data):
        if path == True:
            self.data = pd.read_csv(file_path)
        else:
            self.data = data
        self.summary = None
        self.vif = None
        self.target_freq = None

    def get_summary(self):
        uniques = self.data.nunique()
        dtypes = self.data.dtypes
        missing = self.data.isnull().sum()

        report = pd.DataFrame(uniques)
        report.columns = ["uniques"]
        report["dtypes"] = dtypes
        report["missing"] = missing
        report["missing_pct"] = report.missing / self.data.shape[0]

        self.summary = report

    def categorical(self):
        nunique = self.data.nunique()
        binary_list = nunique[nunique == 2].index.tolist()
        self.data[binary_list] = self.data[binary_list].astype("category")
        # binary_list = self.summary()[self.summary["uniques"] == 2].index.tolist()
        # self.data[binary_list] = self.data[binary_list].astype("category")

        dtypes = self.data.dtypes
        object_list = dtypes[dtypes == "object"].index.tolist()
        # object_list = self.summary()[self.summary()["dtypes"] == "object"].index.tolist()
        self.data[object_list] = self.data[object_list].astype("category")

    def one_hot(self, target):
        y = self.data[target]
        x = self.data.drop(target, axis = 1)
        nunique = x.nunique()
        binary_list = nunique[nunique == 2].index.tolist()

        dtypes = x.dtypes
        object_list = dtypes[dtypes == "object"].index.tolist()

        cat_list = binary_list + object_list
        cat_list = list(set(cat_list))
        x = pd.get_dummies(x, prefix_sep = "_", columns = cat_list, prefix = None)

        self.data = pd.concat([x, y], axis = 1)

        types = self.data.dtypes
        one_hot = types[types == "uint8"].index.tolist()
        self.level_freq = self.data[one_hot].sum(axis = 0)


    def imputation(self, threshold):
        self.get_summary()
        # vars that need imputation
        imput_list = self.summary[(self.summary["missing_pct"] < threshold) & (self.summary["missing_pct"] > 0)]
        imputing = self.data[imput_list.index]

        # vars that don't contain any missings
        no_missing_list = self.summary[self.summary["missing_pct"] == 0]
        no_missing = self.data[no_missing_list.index]

        # impute categorical variables
        imputing_cat = imputing.select_dtypes(exclude="number")
        number_cat = imputing_cat.shape[1]
        if number_cat > 0:
            cat_var = imputing_cat.columns
            cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
            cat_imputted = pd.DataFrame(cat_imputer.fit_transform(imputing_cat))
            cat_imputted.columns = cat_var
            cat_imputted = cat_imputted.astype("category")

        # imputing numerical variables
        imputing_num = imputing.select_dtypes(include="number")
        number_num = imputing_num.shape[1]
        if number_num > 0:
            num_var = imputing_num.columns.tolist()
            num_var_suffix = [x + "_indicator" for x in num_var]
            num_var = num_var + num_var_suffix
            num_imputer = SimpleImputer(strategy="median", add_indicator=True)
            num_imputted = pd.DataFrame(num_imputer.fit_transform(imputing_num))
            num_imputted.columns = num_var
            num_imputted[num_var_suffix] = num_imputted[num_var_suffix].astype("category")

        if number_cat > 0 and number_num > 0:
            imputed_data = pd.concat([cat_imputted, num_imputted], axis=1, sort=False)
            imputed_data = pd.concat([imputed_data, no_missing], axis=1, sort=False)
        elif number_cat > 0 and number_num <= 0:
            imputed_data = pd.concat([cat_imputted, no_missing], axis = 1, sort = False)
        else:
            imputed_data = pd.concat([num_imputted, no_missing], axis = 1, sort = False)

        self.data = imputed_data
        self.get_summary()

    def missing_visualization(self):
        sns.heatmap(self.data.isnull(), cbar=False)

    def multicollinearity(self):
        # Calculating VIF
        nums = self.data._get_numeric_data()

        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif

        nums = nums[vif_list]

        # Cluster the correlation matrix
        Corr = nums.corr()
        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]
        ind_sorted = pd.DataFrame(ind_sorted)
        ind_sorted.columns = ["clusters", "number"]
        ind_sorted["var"] = columns
        freq = ind_sorted["clusters"].value_counts()
        ind_sorted = ind_sorted.merge(freq, how="left", left_on="clusters", right_index=True)
        ind_sorted_noone = ind_sorted[ind_sorted["clusters_y"] != 1]

        # conduct non-parametric ANOVA to decide which variables need to be dropped
        cluster_list = np.unique(ind_sorted_noone["clusters_x"].values)
        drop_list = []
        for i in cluster_list:
            vars = ind_sorted_noone[ind_sorted_noone["clusters_x"] == i]["var"]
            corr = Corr.loc[vars, vars]
            corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
            cluster_num = np.ones(corr.shape[0]) * i
            cluster_num = cluster_num.reshape(corr.shape[0], -1)
            corr = np.concatenate([corr, cluster_num], axis=1)
            corr = pd.DataFrame(corr)
            corr.columns = ["row", "columns", "corr", "clusters"]
            corr = corr[corr["corr"] != 1]
            if corr.shape[0] == 1:
                value = np.array(corr["corr"])
                if value < 0.7:
                    continue
            uniques = np.unique(corr[["row", "columns"]].values)
            p_value = []
            for ii in uniques:
                x = self.data[self.data["TARGET"] == 1][ii]
                y = self.data[self.data["TARGET"] == 0][ii]
                test = stats.kruskal(x, y)
                p_value.append(test[1])

            min = [i for i, j in enumerate(p_value) if j == max(p_value)]
            drop = np.delete(uniques, min)
            for var in drop:
                drop_list.append(var)

        self.data.drop(drop_list, axis = 1, inplace = True)

    def vif_corr_map(self):
        nums = self.data._get_numeric_data()
        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif
        nums = nums[vif_list]
        Corr = nums.corr()

        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]

        nums = nums.reindex(columns, axis = 1)
        Corr = nums.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(Corr, cmap="RdYlBu")
        plt.xticks(range(len(Corr.columns)), Corr.columns, rotation=90)
        plt.yticks(range(len(Corr.columns)), Corr.columns)
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=0.8)

    def get_target_freq(self, target):
        self.target_freq = self.data[target].value_counts()

#%% Data Cleaning
cleaning = Cleaning(False, np.nan, data2)
cleaning.get_summary()
report = cleaning.summary
cleaning.data = cleaning.data[cleaning.data["Type"] == "Cat"]
# cleaning.categorical()
# cleaning.imputation(0.5)

data = cleaning.data.copy()
data.drop("StateName", axis = 1, inplace = True) # drop states
#%% speration check
cats = data.select_dtypes(exclude="number").columns.tolist()
cats.remove("Adopted")

separation = []
for var in cats:
    table = pd.crosstab(index = data["Adopted"], columns = data[var])
    if 0 in table.values:
        separation.append(var)

# we found that there are three variables with potential speration problem: AdoptionSpeed, BreedName_1, and BreedName_2
# AdoptionSpeed
table = pd.crosstab(index = data["Adopted"], columns = data["AdoptionSpeed"])  # AdoptionSpeed is 100% correlated with the target
data.drop("AdoptionSpeed", axis = 1, inplace = True)

# Breed
table = pd.crosstab(index = data["Adopted"], columns = data["BreedName_1"]) # there are too many breed

# combine breed name 1 and breed name 2
cfa_breeds = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair',
              'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla',
              'Chartreux', 'Colorpoint Shorthair', 'Cornish Rex', 'Devon Rex', 'Egyptian Mau', 'European Burmese',
              'Exotic', 'Havana Brown', 'Japanese Bobtail', 'Korat', 'LaPerm', 'Maine Coon', 'Manx',
              'Norwegian Forest Cat', 'Ocicat', 'Oriental', 'Persian', 'Ragamuffin', 'Ragdoll', 'Russian Blue',
              'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Singapura', 'Somali', 'Tonkinese',
              'Turkish Angora', 'Turkish Van']

cfa_breeds = [i.lower() for i in cfa_breeds]
cfa_breeds = '|'.join(cfa_breeds)

data["cfa_1"] = np.where(data["BreedName_1"].str.contains(cfa_breeds, case = False), True, False)
data["cfa_2"] = np.where(data["BreedName_2"].str.contains(cfa_breeds, case = False), True, False)
index = data[(data["cfa_1"] == 0) & (data["cfa_2"] == 1)].index
replace = data.loc[index, :].loc[:, "BreedName_2"].values
data.loc[index, "BreedName_1"] = replace

data["missing_1"] = np.where(data["BreedName_1"].str.contains("missing", case = False), True, False)
data["missing_2"] = np.where(data["BreedName_2"].str.contains("missing", case = False), True, False)
index_missing = data[(data["missing_1"] == 1) & (data["missing_2"] == 0)].index
data.loc[index_missing, "BreedName_1"] = data.loc[index_missing, "BreedName_2"].values

data.drop(["BreedName_2", "cfa_1", "cfa_2", "missing_1", "missing_2"], axis = 1, inplace = True)

data.replace({"BreedName_1": {"tabby": "domestic tobby"}}, inplace = True)
drop_special_breed = data[data.cfa_breeds == 1].BreedName_1.value_counts().iloc[10:].index.to_list()
data.BreedName_1 = data.BreedName_1.replace(to_replace = drop_special_breed, value = "other_special_breed")

# group the breeds that don't belong to any special breeds
drop_breed = data[data.cfa_breeds == 0].BreedName_1.value_counts().iloc[4:].index.to_list()
data.BreedName_1 = data.BreedName_1.replace(to_replace = drop_breed, value = "domestic unknown")

table = pd.crosstab(index = data["Adopted"], columns = data["BreedName_1"])

# State
# table = pd.crosstab(index = data["Adopted"], columns = data["StateName"]) # Lubuan only has one record, drop it to avoid seperation
# data = data[data.StateName != "Labuan"]

#%% Data preprocessing
data.drop(["Quantity", "ColorName_2", "ColorName_3", "BreedName_1"], axis = 1, inplace = True)  # drop unnecessary column
data.replace({"Adopted": {"Yes": 1, "NoAfter100": 0}}, inplace = True)
# data.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/logistic.csv", index = False)

# vif_corr
nums = data._get_numeric_data()
nums = nums.iloc[:, :4]
vif = pd.DataFrame()
vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
vif["features"] = nums.columns
vif_list = vif[vif["factor"] >= 5]["features"]

# reference coding
vars = data.columns.tolist()
vars.remove("Adopted")
vars = " + ".join(vars)
f = "Adopted ~ " + vars
y, x = patsy.dmatrices(formula_like = f, data = data, NA_action = "raise", return_type = "dataframe")
# y = y.iloc[:, 1]

# training and testing split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=55)

# standardize data
scaler = preprocessing.StandardScaler()

x_train_cont = x_train.iloc[:, -4:]
x_test_cont = x_test.iloc[:, -4:]
x_train_scaled = scaler.fit_transform(x_train_cont)
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train_cont.columns)
x_test_scaled = scaler.transform(x_test_cont)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test_cont.columns)

x_train_cat = x_train.iloc[:, :-4]
x_test_cat = x_test.iloc[:, :-4]

x_train_stand = pd.concat([x_train_cat.reset_index(drop = True), x_train_scaled], axis = 1)
x_test_stand = pd.concat([x_test_cat.reset_index(drop = True), x_test_scaled], axis = 1)

x_train_stand.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/x_train.csv", index = False)
x_test_stand.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/x_test.csv", index = False)
y_train.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/y_train.csv", index = False)
y_test.to_csv("/Users/dauku/Desktop/Python/AnimalShelter/petfinder-adoption-prediction/y_test.csv", index = False)

x = x_train_stand.append(x_test_stand, ignore_index=True)
y = y_train.append(y_test, ignore_index = True)

#%% initial model building (LASSO logistic regression)
logistic_regression = Logit(y_train.values, x_train_stand)
alpha = np.linspace(0, 1000, 101)
auc = []
for a in alpha:
    rslt = logistic_regression.fit_regularized(alpha = a, disp = False)
    prediction = rslt.predict(exog = x_test_stand)
    auc.append(roc_auc_score(y_test, prediction))
auc = np.array(auc)

# 0 alpha gives the best auc, therefore we can use the regular logistic regression
logistic_result = logistic_regression.fit()
logistic_prediction = logistic_result.predict(exog = x_test_stand)
logistic_result.summary()
auc_score = round(roc_auc_score(y_test, logistic_prediction), 2)

#%% ROC curve
def ROC(true, prediction, model):
    y_test = true
    prediction = prediction
    fpr, tpr, t = roc_curve(y_true = y_test, y_score = prediction)
    auc_score = round(roc_auc_score(y_test, prediction), 2)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label=model)
    plt.plot([0, 1], [0, 1], color='k', linestyle='-', linewidth=2)
    plt.text(0.8, 0.2, f"AUC: {auc_score}", bbox=dict(facecolor='red', alpha=0.2))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

ROC(y_test, logistic_prediction, "Logistic Regression")

#%% random forest
random_forest = RandomForestClassifier(n_estimators = 250, criterion = "gini", class_weight = "balanced_subsample",
                                       bootstrap = True, oob_score = True)
random_forest.fit(x_train_stand, y_train.values.flatten())

feature_importance = random_forest.feature_importances_
importance = pd.DataFrame({"importance": feature_importance, "vars": x_train_stand.columns}).sort_values(by = "importance",
                                                                                                         ascending = False)

plt.figure(figsize = (15, 5))
plt.xticks(rotation = 90)
sns.barplot(x = "vars", y = "importance", data = importance[0:15])
plt.tight_layout()
plt.show()

# prediction
prediction_rf = random_forest.predict_proba(x_test)[:, 1]
auc_score_rf = round(roc_auc_score(y_test, prediction_rf), 2)
ROC(y_test, prediction_rf, "Initial Random Forest")

#%% tune random forest
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=55, n_jobs = -1, scoring = "roc_auc")
# Fit the random search model
rf_random.fit(x, y.values.flatten())

rf_random.best_params_
rf_random.best_score_
best_rf = RandomForestClassifier(n_estimators = 2000, criterion = "gini", min_samples_split = 2,
                                       bootstrap = False, min_samples_leaf = 2, max_features = "auto",
                                 max_depth = 10, random_state = 55)


best_rf.fit(x_train_stand, y_train.values.flatten())
best_prediction = best_rf.predict_proba(x_test_stand)[:, 1]
roc_auc_score(y_test, best_prediction)

ROC(y_test, best_prediction[:, 1], "Tuned Random Forest")

feature_importance = best_rf.feature_importances_
importance = pd.DataFrame({"importance": feature_importance, "vars": x_train_stand.columns}).sort_values(by = "importance",
                                                                                                         ascending = False)

plt.figure(figsize = (15, 5))
plt.xticks(rotation = 90)
sns.barplot(x = "vars", y = "importance", data = importance[0:6])
plt.tight_layout()
plt.show()
# def custom_auc(ground_truth, predictions):
#     # I need only one column of predictions["0" and "1"]. You can get an error here
#     # while trying to return both columns at once
#     fpr, tpr, t = roc_curve(y_true=ground_truth, y_score=predictions)
#     auc_score = roc_auc_score(ground_truth, predictions)
#     return auc_score
#
# my_auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)
#%% plot three ROC curves together
fpr_logit, tpr_logit, t = roc_curve(y_true=y_test, y_score=logistic_prediction)
fpr_rf_i, tpr_rf_i, t_rf_i = roc_curve(y_true=y_test, y_score=prediction_rf)
fpr_rf_t, tpr_rf_t, t_rf_t = roc_curve(y_true=y_test, y_score=best_prediction)

auc_score_logit = round(roc_auc_score(y_test, logistic_prediction), 2)
auc_score_rf_i = round(roc_auc_score(y_test, prediction_rf), 2)
auc_score_rf_t = round(roc_auc_score(y_test, best_prediction), 2)

# plot the roc curve for the model
plt.plot(fpr_logit, tpr_logit, marker='.', label=f"Logistic Regression: {auc_score_logit}", color = "b")
plt.plot(fpr_rf_i, tpr_rf_i, marker='.', label=f"Initial Random Forest: {auc_score_rf_i}", color = "r")
plt.plot(fpr_rf_t, tpr_rf_t, marker='.', label=f"Tuned Random Forest: {auc_score_rf_t}", color = "g")

plt.plot([0, 1], [0, 1], color='k', linestyle='-', linewidth=2)
# plt.text(0.8, 0.2, f"AUC: {auc_score}", bbox=dict(facecolor='red', alpha=0.2))

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()