#%%
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
import patsy

#%%
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

#%%
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
table = pd.crosstab(index = data["Adopted"], columns = data["StateName"]) # Lubuan only has one record, drop it to avoid seperation
data = data[data.StateName != "Labuan"]

#%% initial model building (random forest)
data.drop(["Quantity"], axis = 1, inplace = True)  # drop unnecessary column

