import csv
import h5py
import io
import numpy as np
import pandas as pd
import pkgutil
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

EPS = 1e-8

def load_dataset(dataset, random_seed_offset=0, test_size=0.3,
                 fix_test_shuffle_train=False, competing=True,
                 time_series=False):
    """
    Loads a dataset.

    Parameters
    ----------
    dataset : string
        One of 'pbc', 'framingham', 'seer'.

        Note that a dataset could come with its own pre-specified training
        and test data (for example, for 'rotterdam-gbsg', we train
        on the Rotterdam tumor bank data and test on GBSG2 data). If the
        dataset does *not* come with pre-specified training and test data
        (such as for 'support'), then the code will generate a "random"
        train/test split ("random" is in quotes since to make the code
        reproducible, we have seeded the randomness per dataset with a fixed
        integer, so that the code should produce the same random train/test
        splits across experimental repeats when
        `random_seed_offset` is set to the same value).

    random_seed_offset : int, optional (default=0)
        Offset to add to random seed in shuffling the data.

    test_size: float or int, optional (default=0.3)
        If specified as a float, then this is the fraction of data to treat as
        the test data. If specified as an int, then this is the number of
        points to treat as the test data.

    fix_test_shuffle_train: boolean (default=False)
        If this flag is set to be True, then the test data will be treated as
        fixed meaning that using different values of `random_seed_offset` lead
        to different shuffled versions of the training data but the test data
        remain the same.

        If instead this flag is set to be False, then we actually shuffle the
        full dataset prior to making the train/test split. Thus, different
        values of `random_seed_offset` lead to different train/test splits
        (so which points end up in training vs test data varies across
        different values of `random_seed_offset`).

    competing: boolean (default=False)
        If this flag is set to be True, then the training and test set labels
        will use event indicators that specify which critical event happened
        (where 0 means censoring happened prior to any critical event).
        
        Note that if the dataset does not actually support competing risks,
        then the value of this flag does not affect anything (basically we
        would just be using the competing risks setup with a single critical
        event).

    time_series: boolean (default=False)
        For datasets that are not actually time series, this flag should be
        left as the default value of False (and setting this flag to be True
        for such datasets will result in an error).

        For datasets that are time series, if this flag is set to be False,
        then we only take the first time step per time series, and we thus
        treat the dataset as if it were a regular tabular dataset. If
        instead, this flag is set to be True for a time series dataset,
        then the training and test data can be variable in length.

    Returns
    -------
    X_train : either a 2D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 2D numpy array with shape = [n_samples, n_features].
        This array will consist of the training feature vectors.

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series represented as a 2D numpy array with
        shape [n_time_steps, n_features]. Different data points could have
        different numbers of time steps.

    Y_train : either a 1D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 1D numpy array with length = n_samples. This array
        will consist of training observed times (in the same order as the
        rows of `X_train`).

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series's observed times across time, represented as
        a 1D numpy array with length = n_time_steps.

    D_train : either a 1D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 1D numpy array with length = n_samples. This array
        will consist of training event indicators (in the same order as the
        rows of `X_train`).

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series's event indicators across time (which
        commonly does not actually change), represented as a 1D numpy array
        with length = n_time_steps.

    X_test : either a 2D numpy array or a list
        Same format as `X_train` but now for test data.

    Y_test : either a 1D numpy array or a list
        Same format as `Y_train` but now for test data.

    D_test : either a 1D numpy array or a list
        Same format as `D_train` but now for test data.

    features_before_preprocessing : list
        List of strings specifying the names of the features *before* applying
        preprocessing.

    features_after_preprocessing : list
        List of strings specifying the names of the features *after* applying
        preprocessing.

    events : list
        List of strings specifying the event names. For standard survival
        analysis datasets (without competing risks), this list will consist of
        a single element. When there are competing risks, the length of this
        list is the number of competing risks.

    train_test_split_prespecified : boolean
        If True, then this means that the dataset comes with its own train/test
        split and therefore there is no randomization for this split.

    build_preprocessor_and_preprocess : function
        Function for fitting and then preprocessing features into some
        "standardized"/"normalized" feature space. This should be applied to
        training feature vectors prior to using a learning algorithm (unless the
        learning algorithm does not need this sort of normalization). This
        function returns both the normalized features and a preprocessor object
        (see the next output for how to use this preprocessor object).

    apply_preprocessor : function
        Function that, given feature vectors (e.g., validation/test data) and a
        preprocessor object (created via `build_preprocessor_and_preprocess`),
        preprocesses the feature vectors as to put them in a normalized feature
        space.
    """
    
    if dataset.startswith('sync') and 'train' in dataset:
        diff_train_size = True
        train_data_subset_rate = float(dataset[4]) / float(dataset[6])
        dataset = 'synthetic'
    else:
        diff_train_size = False
        

    if dataset == 'pbc':
        df = pd.read_csv('data/pbc2.csv').astype({'edema': 'category'})

        def map_yes_no_or_missing_to_number(x):
            if type(x) != str:
                return np.nan
            elif x == 'Yes':
                return 1.0
            else:
                return 0.0

        df['drug'] = df['drug'].apply(lambda x: 1*(x == 'D-penicil'))  # no nan
        df['sex'] = df['sex'].apply(lambda x: 1*(x == 'female'))  # no nan
        df['ascites'] = df['ascites'].apply(map_yes_no_or_missing_to_number)
        df['hepatomegaly'] = \
            df['hepatomegaly'].apply(map_yes_no_or_missing_to_number)
        df['spiders'] = df['spiders'].apply(map_yes_no_or_missing_to_number)
        df = df.rename(columns={'sex': 'female', 'drug': 'D-penicil'})
        df['age'] = df['age'] + df['years']
        edema_categories = df['edema'].cat.categories.to_list()
        df['edema'] = df['edema'].cat.codes

        features_before_preprocessing = \
            ['D-penicil', 'female', 'ascites', 'hepatomegaly', 'spiders',
             'edema', 'histologic', 'serBilir', 'serChol', 'albumin',
             'alkaline', 'SGOT', 'platelets', 'prothrombin', 'age']

        features_after_preprocessing = \
            ['D-penicil', 'female', 'ascites', 'hepatomegaly', 'spiders',
             'histologic_norm', 'serBilir_std', 'serChol_std', 'albumin_std',
             'alkaline_std', 'SGOT_std', 'platelets_std', 'prothrombin_std',
             'age_std', 'edema_no', 'edema_yes_despite_diuretics',
             'edema_yes_without_diuretics']

        features = df[features_before_preprocessing].to_numpy().astype('float32')
        observed_times = (df['years'] - df['year']).to_numpy().astype('float32')
        event_indicators = (df['status'] == 'dead').to_numpy().astype('int32')
        events = ['death']
        if competing:
            event_indicators[df['status'] == 'transplanted'] = 2
            events.append('transplanted')

        if not time_series:
            X = features
            Y = observed_times
            D = event_indicators
        else:
            X = []
            Y = []
            D = []
            for id in sorted(list(set(df['id']))):
                mask = (df['id'] == id)
                X.append(features[mask])
                Y.append(observed_times[mask])
                D.append(event_indicators[mask])

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
               0: D-penicil    (we will leave the same; originally a binary indicator)
               1: female       (we will leave the same; originally a binary indicator)
               2: ascites      (we will leave the same; originally a binary indicator)
               3: hepatomegaly (we will leave the same; originally a binary indicator)
               4: spiders      (we will leave the same; originally a binary indicator)
               5: edema        (we will treat as categorical and one-hot encode)
               6: histologic   (we will subtract 1 and then divide by 3)
               7: serBilir     (we will standardize)
               8: serChol      (we will standardize)
               9: albumin      (we will standardize)
              10: alkaline     (we will standardize)
              11: SGOT         (we will standardize)
              12: platelets    (we will standardize)
              13: prothrombin  (we will standardize)
              14: age          (we will standardize)

            After preprocessing, the new features are:
               0: D-penicil
               1: female
               2: ascites
               3: hepatomegaly
               4: spiders
               5: (histologic - 1) / 3
               6: standardized serBilir
               7: standardized serChol
               8: standardized albumin
               9: standardized alkaline
              10: standardized SGOT
              11: standardized platelets
              12: standardized prothrombin
              13: standardized age
              14: edema = no
              15: edema = yes despite diuretics
              16: edema = yes without diuretics

            When a Cox model is fitted (argument `cox` is set to True),
            then we drop feature 18 treating it as the reference value
            for categorical variable `race` as to avoid collinearity
            (this sort of preprocessing is standard for working with
            Cox models and categorical variables where one category
            is treated as the reference or baseline value and is omitted).
            """
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 17))
            imputer = SimpleImputer(missing_values=np.nan,
                                    strategy='mean')
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=[list(range(3))])
            features_imputed = imputer.fit_transform(features_stacked)
            cols_standardize = [7, 8, 9, 10, 11, 12, 13, 14]
            cols_leave = [0, 1, 2, 3, 4]
            cols_categorical = [5]
            new_features[:, [0, 1, 2, 3, 4]] \
                = features_imputed[:, cols_leave]
            new_features[:, 5] = (features_imputed[:, 6] - 1) / 3.
            new_features[:, [6, 7, 8, 9, 10, 11, 12, 13]] = \
                scaler.fit_transform(
                    features_imputed[:, cols_standardize])
            new_features[:, 14:] = \
                encoder.fit_transform(
                    features_imputed[:, cols_categorical]).toarray()
            if cox:
                new_features = new_features[:, :-1]

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked, (imputer, scaler, encoder)
            else:
                return new_features, (imputer, scaler, encoder)

        def apply_preprocessor(features, preprocessor, cox=False):
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 17))
            imputer, scaler, encoder = preprocessor
            features_imputed = imputer.transform(features_stacked)
            cols_standardize = [7, 8, 9, 10, 11, 12, 13, 14]
            cols_leave = [0, 1, 2, 3, 4]
            cols_categorical = [5]
            new_features[:, [0, 1, 2, 3, 4]] \
                = features_imputed[:, cols_leave]
            new_features[:, 5] = (features_imputed[:, 6] - 1) / 3.
            new_features[:, [6, 7, 8, 9, 10, 11, 12, 13]] = \
                scaler.transform(
                    features_imputed[:, cols_standardize])
            new_features[:, 14:] = \
                encoder.transform(
                    features_imputed[:, cols_categorical]).toarray()
            if cox:
                new_features = new_features[:, :-1]

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked
            else:
                return new_features

        dataset_random_seed = 2893429804
        train_test_split_prespecified = False


    elif dataset == 'framingham':
        # The following code is cited from the below link with minor changes:
        # https://github.com/Jeanselme/DeepSurvivalMachines/blob/e4b07b3f497f2266eaa71d0e182195e95663d367/dsm/datasets.py#L57
        df = pd.read_csv('data/framingham.csv')
        if not time_series:
            # Consider only first event
            df = df.groupby('RANDID').first()

        binary_col = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
                        'PREVCHD', 'PREVAP', 'PREVMI',
                        'PREVSTRK', 'PREVHYP']
        cat_not_binary_col = ['educ']
        num_col = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP',
                   'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
        df['SEX'] = df['SEX'] - 1 # map 1,2 to 0,1 (it has no missing values)
        features_before_preprocessing = binary_col + cat_not_binary_col + num_col
        features_after_preprocessing = binary_col + ['educ_1', 'educ_2', 'educ_3', 'educ_4'] + [f'{col}_std' for col in num_col]

        features = df[features_before_preprocessing].to_numpy().astype('float32')
        observed_times = (df['TIMEDTH'] - df['TIME']).to_numpy().astype('float32')
        event_indicators = df['DEATH'].to_numpy().astype('int32')
        events = ['death']
        if competing:
            time_cvd = (df['TIMECVD'] - df['TIME']).values
            event_indicators[df['CVD'] == 1] = 2
            observed_times[df['CVD'] == 1] = time_cvd[df['CVD'] == 1]
            events.append('CVD')

        # time = (df['TIMEDTH'] - df['TIME']).values
        # event = df['DEATH'].values
        
        # if competing:
        #     time_cvd = (df['TIMECVD'] - df['TIME']).values
        #     event[df['CVD'] == 1] = 2
        #     time[df['CVD'] == 1] = time_cvd[df['CVD'] == 1]
        #     events.append('CVD')

        # x_ = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)

        if not time_series:
            X = features
            Y = observed_times + 1
            D = event_indicators
            # return x_, time + 1, event, np.concatenate([x1.columns, x2.columns])
        else:
            X = []
            Y = []
            D = []
            mask = observed_times > 0
            features, df, observed_times, event_indicators = features[mask], df[mask], observed_times[mask], event_indicators[mask]
            # x_, df, time, event = x_[time > 0], df[time > 0], time[time > 0], event[time > 0]
            # x, t, e = [], [], []
            for id_ in sorted(list(set(df['RANDID']))):
                mask_id = df['RANDID'] == id_
                X.append(features[mask_id])
                Y.append(observed_times[mask_id] + 1)
                D.append(event_indicators[mask_id])
            # return x, t, e, np.concatenate([x1.columns, x2.columns])

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
                0: SEX        (we will leave the same; originally a binary indicator after doing SEX = SEX - 1)
                1: CURSMOKE   (we will leave the same; originally a binary indicator)
                2: DIABETES   (we will leave the same; originally a binary indicator)
                3: BPMEDS     (we will leave the same; originally a binary indicator)
                4: PREVCHD    (we will leave the same; originally a binary indicator)
                5: PREVAP     (we will leave the same; originally a binary indicator)
                6: PREVMI     (we will leave the same; originally a binary indicator)
                7: PREVSTRK   (we will leave the same; originally a binary indicator)
                8: PREVHYP    (we will leave the same; originally a binary indicator)
                9: educ       (we will treat as categorical and one-hot encode)
                10: TOTCHOL   (we will standardize)
                11: AGE       (we will standardize)
                12: SYSBP     (we will standardize)
                13: DIABP     (we will standardize)
                14: CIGPDAY   (we will standardize)
                15: BMI       (we will standardize)
                16: HEARTRTE  (we will standardize)
                17: GLUCOSE   (we will standardize)

            After preprocessing, the new features are:
                0: SEX
                1: CURSMOKE
                2: DIABETES
                3: BPMEDS
                4: PREVCHD
                5: PREVAP
                6: PREVMI
                7: PREVSTRK
                8: PREVHYP
                9: educ_1
                10: educ_2
                11: educ_3
                12: educ_4
                13: standardized TOTCHOL
                14: standardized AGE
                15: standardized SYSBP
                16: standardized DIABP
                17: standardized CIGPDAY
                18: standardized BMI
                19: standardized HEARTRTE
                20: standardized GLUCOSE
            """
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 21))

            cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=[[1,2,3,4]])

            cat_features_imputed = cat_imputer.fit_transform(features_stacked[:, :10])
            num_features_imputed = num_imputer.fit_transform(features_stacked[:, 10:])
            new_features[:, :9] = cat_features_imputed[:, :9] # no standardization binary features
            new_features[:, 9:13] = encoder.fit_transform(cat_features_imputed[:, 9:10].astype('int32')).toarray()
            new_features[:, 13:] = scaler.fit_transform(num_features_imputed)

            if cox:
                # if cox == 'csCox':
                #     new_features = np.delete(new_features, [5,7,12], axis=1)
                # else:
                new_features = np.delete(new_features, 12, axis=1)

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked, (cat_imputer, num_imputer, scaler, encoder)
            else:
                return new_features, (cat_imputer, num_imputer, scaler, encoder)
            
        def apply_preprocessor(features, preprocessor, cox=False):
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 21))
            cat_imputer, num_imputer, scaler, encoder = preprocessor

            cat_features_imputed = cat_imputer.transform(features_stacked[:, :10])
            num_features_imputed = num_imputer.transform(features_stacked[:, 10:])
            new_features[:, :9] = cat_features_imputed[:, :9] # no standardization binary features
            new_features[:, 9:13] = encoder.transform(cat_features_imputed[:, 9:10].astype('int32')).toarray()
            new_features[:, 13:] = scaler.transform(num_features_imputed)

            if cox:
                # if cox == 'csCox':
                #     new_features = np.delete(new_features, [5,7,12], axis=1)
                # else:
                new_features = np.delete(new_features, 12, axis=1)

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked
            else:
                return new_features
            
        dataset_random_seed = 541166515
        train_test_split_prespecified = False
        
    
    elif dataset == 'seer2010':
        if not competing:
            raise ValueError(f'For seer dataset, we currently only support competing=True, but got competing={competing}.')
        if time_series:
            raise ValueError(f'For seer dataset, we currently only support time_series=False, but got time_series={time_series}.')
        
        # The following code is cited from the below link with minor changes:
        # https://github.com/Jeanselme/NeuralFineGray/blob/main/nfg/datasets.py#L39
        df = pd.read_csv('data/seer_test (trying to match vincent, with selection, fixed grade, nov 2024 sub).csv')
        # restrict to year of 2010
        df = df[df['Year of diagnosis'] == 2010]
        # Remove multiple visits
        df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008'])

        # Grade Recode (thru 2017)
        df["Grade Recode (thru 2017)"].replace('Well differentiated; Grade I', '1', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Moderately differentiated; Grade II', '2', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Poorly differentiated; Grade III', '3', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Undifferentiated; anaplastic; Grade IV', '4', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Unknown', '9', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Blank(s)', 'NaN', inplace = True)

        # Encode using dictionary to remove missing data
        df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)
        df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
        df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
        df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
        df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

        # Remove patients without survival time
        df = df[~df.duration.isna()]

        # Outcome
        df['duration'] = df['duration'].astype(float)
        df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)" # Death
        df.loc[(df["COD to site recode"] == "Diseases of Heart") & \
               (df["SEER cause-specific death classification"] == "Alive or dead of other cause"), 'event'] = 2 # CVD

        df = df.drop(columns = ["COD to site recode"])

        ## Categorical (Modified by anonymous authors):
        #   ["Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant"] are duplicated features when we only look at malignant cancer cohort.
        #   Both of them are removed as they are too granular,  while SEER dataset provides "Histology recode - broad groupings", 
        #   which map Histologic Type into broader categories for easier analysis.
        categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality",
                "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
                "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
                "Sequence number", "RX Summ--Surg Prim Site (1998+)",
                "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", 
                "Origin recode NHIA (Hispanic, Non-Hisp)", "Grade Recode (thru 2017)"]
        ordinal_col = ["Age recode with <1 year olds", "Year of diagnosis"]
        ## Numerical
        numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
            "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
        

        df[ordinal_col] = df[ordinal_col].replace(
        {age: number
            for number, age in enumerate(['01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
            '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years',
            '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
        }).replace({
            grade: number
            for number, grade in enumerate(['1', '2', '3', '4'])
        })

        cat_imputer = SimpleImputer(strategy='most_frequent')
        ord_enc = OrdinalEncoder()
        df[categorical_col] = ord_enc.fit_transform(cat_imputer.fit_transform(df[categorical_col]).astype(str))
        df[ordinal_col] = cat_imputer.fit_transform(df[ordinal_col])
        
        number_of_cat_per_cat_col = dict(df[categorical_col].nunique())
        cat_features_after_preprocessing = [f'{col}: {ord_enc.categories_[col_idx][cat_idx]}' for col_idx, col in enumerate(categorical_col) for cat_idx in range(number_of_cat_per_cat_col[col])]
        
        features_before_preprocessing = categorical_col + ordinal_col + numerical_col
        features_after_preprocessing = \
            cat_features_after_preprocessing + \
                ["Age_group", "Year of diagnosis_std"] + \
                [f"{col}_std" for col in numerical_col]
        
        features = df[features_before_preprocessing].to_numpy()
        observed_times = df['duration'].to_numpy().astype('float32')
        event_indicators = df['event'].to_numpy().astype('float32')
        events = ['BC', 'CVD']

        X = features
        Y = observed_times
        D = event_indicators

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
                0: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)
                1: Laterality
                2: Diagnostic Confirmation
                3: Histology recode - broad groupings
                4: Chemotherapy recode (yes, no/unk)
                5: Radiation recode
                6: ER Status Recode Breast Cancer (1990+)
                7: PR Status Recode Breast Cancer (1990+)
                8: Sequence number
                9: RX Summ--Surg Prim Site (1998+)
                10: CS extension (2004-2015)
                11: CS lymph nodes (2004-2015)
                12: CS mets at dx (2004-2015)
                13: Origin recode NHIA (Hispanic, Non-Hisp)
                14: Grade Recode (thru 2017)
                15: Age recode with <1 year olds
                16: Year of diagnosis
                17: Total number of in situ/malignant tumors for patient
                18: Total number of benign/borderline tumors for patient
                19: CS tumor size (2004-2015)
                20: Regional nodes examined (1988+)
                21: Regional nodes positive (1988+)

            After preprocessing, the new features are:
                0: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Hispanic (All Races)
                1: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic American Indian/Alaska Native
                2: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Asian or Pacific Islander
                3: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Black
                4: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Unknown Race
                5: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic White
                6: Laterality: Bilateral, single primary
                7: Laterality: Left - origin of primary
                8: Laterality: Only one side - side unspecified
                9: Laterality: Paired site, but no information concerning laterality
                10: Laterality: Right - origin of primary
                11: Diagnostic Confirmation: Clinical diagnosis only
                12: Diagnostic Confirmation: Direct visualization without microscopic confirmation
                13: Diagnostic Confirmation: Positive exfoliative cytology, no positive histology
                14: Diagnostic Confirmation: Positive histology
                15: Diagnostic Confirmation: Positive microscopic confirm, method not specified
                16: Diagnostic Confirmation: Radiography without microscopic confirm
                17: Histology recode - broad groupings: 8000-8009: unspecified neoplasms
                18: Histology recode - broad groupings: 8010-8049: epithelial neoplasms, NOS
                19: Histology recode - broad groupings: 8050-8089: squamous cell neoplasms
                20: Histology recode - broad groupings: 8140-8389: adenomas and adenocarcinomas
                21: Histology recode - broad groupings: 8390-8429: adnexal and skin appendage neoplasms
                22: Histology recode - broad groupings: 8430-8439: mucoepidermoid neoplasms
                23: Histology recode - broad groupings: 8440-8499: cystic, mucinous and serous neoplasms
                24: Histology recode - broad groupings: 8500-8549: ductal and lobular neoplasms
                25: Histology recode - broad groupings: 8560-8579: complex epithelial neoplasms
                26: Histology recode - broad groupings: 8800-8809: soft tissue tumors and sarcomas, NOS
                27: Histology recode - broad groupings: 8810-8839: fibromatous neoplasms
                28: Histology recode - broad groupings: 8850-8889: lipomatous neoplasms
                29: Histology recode - broad groupings: 8890-8929: myomatous neoplasms
                30: Histology recode - broad groupings: 8930-8999: complex mixed and stromal neoplasms
                31: Histology recode - broad groupings: 9000-9039: fibroepithelial neoplasms
                32: Histology recode - broad groupings: 9120-9169: blood vessel tumors
                33: Histology recode - broad groupings: 9180-9249: osseous and chondromatous neoplasms
                34: Histology recode - broad groupings: 9260-9269: miscellaneous bone tumors (C40._, C41._)
                35: Chemotherapy recode (yes, no/unk): No/Unknown
                36: Chemotherapy recode (yes, no/unk): Yes
                37: Radiation recode: Beam radiation
                38: Radiation recode: Combination of beam with implants or isotopes
                39: Radiation recode: None/Unknown
                40: Radiation recode: Radiation, NOS  method or source not specified
                41: Radiation recode: Radioactive implants (includes brachytherapy) (1988+)
                42: Radiation recode: Radioisotopes (1988+)
                43: Radiation recode: Recommended, unknown if administered
                44: Radiation recode: Refused (1988+)
                45: ER Status Recode Breast Cancer (1990+): Borderline/Unknown
                46: ER Status Recode Breast Cancer (1990+): Negative
                47: ER Status Recode Breast Cancer (1990+): Positive
                48: ER Status Recode Breast Cancer (1990+): Recode not available
                49: PR Status Recode Breast Cancer (1990+): Borderline/Unknown
                50: PR Status Recode Breast Cancer (1990+): Negative
                51: PR Status Recode Breast Cancer (1990+): Positive
                52: PR Status Recode Breast Cancer (1990+): Recode not available
                53: Sequence number: 1st of 2 or more primaries
                54: Sequence number: 2nd of 2 or more primaries
                55: Sequence number: 3rd of 3 or more primaries
                56: Sequence number: 4th of 4 or more primaries
                57: Sequence number: 5th of 5 or more primaries
                58: Sequence number: 6th of 6 or more primaries
                59: Sequence number: One primary only
                60: RX Summ--Surg Prim Site (1998+): 00
                61: RX Summ--Surg Prim Site (1998+): 19
                62: RX Summ--Surg Prim Site (1998+): 20
                63: RX Summ--Surg Prim Site (1998+): 21
                64: RX Summ--Surg Prim Site (1998+): 22
                65: RX Summ--Surg Prim Site (1998+): 23
                66: RX Summ--Surg Prim Site (1998+): 24
                67: RX Summ--Surg Prim Site (1998+): 30
                68: RX Summ--Surg Prim Site (1998+): 40
                69: RX Summ--Surg Prim Site (1998+): 41
                70: RX Summ--Surg Prim Site (1998+): 42
                71: RX Summ--Surg Prim Site (1998+): 43
                72: RX Summ--Surg Prim Site (1998+): 44
                73: RX Summ--Surg Prim Site (1998+): 45
                74: RX Summ--Surg Prim Site (1998+): 46
                75: RX Summ--Surg Prim Site (1998+): 47
                76: RX Summ--Surg Prim Site (1998+): 48
                77: RX Summ--Surg Prim Site (1998+): 49
                78: RX Summ--Surg Prim Site (1998+): 50
                79: RX Summ--Surg Prim Site (1998+): 51
                80: RX Summ--Surg Prim Site (1998+): 52
                81: RX Summ--Surg Prim Site (1998+): 53
                82: RX Summ--Surg Prim Site (1998+): 54
                83: RX Summ--Surg Prim Site (1998+): 55
                84: RX Summ--Surg Prim Site (1998+): 56
                85: RX Summ--Surg Prim Site (1998+): 57
                86: RX Summ--Surg Prim Site (1998+): 58
                87: RX Summ--Surg Prim Site (1998+): 59
                88: RX Summ--Surg Prim Site (1998+): 60
                89: RX Summ--Surg Prim Site (1998+): 61
                90: RX Summ--Surg Prim Site (1998+): 62
                91: RX Summ--Surg Prim Site (1998+): 63
                92: RX Summ--Surg Prim Site (1998+): 64
                93: RX Summ--Surg Prim Site (1998+): 65
                94: RX Summ--Surg Prim Site (1998+): 66
                95: RX Summ--Surg Prim Site (1998+): 67
                96: RX Summ--Surg Prim Site (1998+): 68
                97: RX Summ--Surg Prim Site (1998+): 69
                98: RX Summ--Surg Prim Site (1998+): 70
                99: RX Summ--Surg Prim Site (1998+): 71
                100: RX Summ--Surg Prim Site (1998+): 72
                101: RX Summ--Surg Prim Site (1998+): 73
                102: RX Summ--Surg Prim Site (1998+): 74
                103: RX Summ--Surg Prim Site (1998+): 75
                104: RX Summ--Surg Prim Site (1998+): 76
                105: RX Summ--Surg Prim Site (1998+): 80
                106: RX Summ--Surg Prim Site (1998+): 90
                107: RX Summ--Surg Prim Site (1998+): 99
                108: CS extension (2004-2015): 050
                109: CS extension (2004-2015): 070
                110: CS extension (2004-2015): 100
                111: CS extension (2004-2015): 110
                112: CS extension (2004-2015): 120
                113: CS extension (2004-2015): 130
                114: CS extension (2004-2015): 140
                115: CS extension (2004-2015): 170
                116: CS extension (2004-2015): 180
                117: CS extension (2004-2015): 190
                118: CS extension (2004-2015): 200
                119: CS extension (2004-2015): 300
                120: CS extension (2004-2015): 400
                121: CS extension (2004-2015): 410
                122: CS extension (2004-2015): 512
                123: CS extension (2004-2015): 514
                124: CS extension (2004-2015): 516
                125: CS extension (2004-2015): 518
                126: CS extension (2004-2015): 519
                127: CS extension (2004-2015): 520
                128: CS extension (2004-2015): 575
                129: CS extension (2004-2015): 580
                130: CS extension (2004-2015): 585
                131: CS extension (2004-2015): 600
                132: CS extension (2004-2015): 605
                133: CS extension (2004-2015): 612
                134: CS extension (2004-2015): 615
                135: CS extension (2004-2015): 680
                136: CS extension (2004-2015): 725
                137: CS extension (2004-2015): 730
                138: CS extension (2004-2015): 750
                139: CS extension (2004-2015): 780
                140: CS extension (2004-2015): 790
                141: CS extension (2004-2015): 950
                142: CS extension (2004-2015): 999
                143: CS lymph nodes (2004-2015): 000
                144: CS lymph nodes (2004-2015): 050
                145: CS lymph nodes (2004-2015): 130
                146: CS lymph nodes (2004-2015): 150
                147: CS lymph nodes (2004-2015): 155
                148: CS lymph nodes (2004-2015): 250
                149: CS lymph nodes (2004-2015): 255
                150: CS lymph nodes (2004-2015): 257
                151: CS lymph nodes (2004-2015): 258
                152: CS lymph nodes (2004-2015): 260
                153: CS lymph nodes (2004-2015): 510
                154: CS lymph nodes (2004-2015): 520
                155: CS lymph nodes (2004-2015): 600
                156: CS lymph nodes (2004-2015): 610
                157: CS lymph nodes (2004-2015): 620
                158: CS lymph nodes (2004-2015): 630
                159: CS lymph nodes (2004-2015): 710
                160: CS lymph nodes (2004-2015): 720
                161: CS lymph nodes (2004-2015): 740
                162: CS lymph nodes (2004-2015): 745
                163: CS lymph nodes (2004-2015): 750
                164: CS lymph nodes (2004-2015): 755
                165: CS lymph nodes (2004-2015): 760
                166: CS lymph nodes (2004-2015): 763
                167: CS lymph nodes (2004-2015): 765
                168: CS lymph nodes (2004-2015): 768
                169: CS lymph nodes (2004-2015): 800
                170: CS lymph nodes (2004-2015): 805
                171: CS lymph nodes (2004-2015): 810
                172: CS lymph nodes (2004-2015): 815
                173: CS lymph nodes (2004-2015): 820
                174: CS lymph nodes (2004-2015): 999
                175: CS mets at dx (2004-2015): 00
                176: CS mets at dx (2004-2015): 05
                177: CS mets at dx (2004-2015): 07
                178: CS mets at dx (2004-2015): 10
                179: CS mets at dx (2004-2015): 40
                180: CS mets at dx (2004-2015): 42
                181: CS mets at dx (2004-2015): 44
                182: CS mets at dx (2004-2015): 50
                183: CS mets at dx (2004-2015): 60
                184: CS mets at dx (2004-2015): 99
                185: Origin recode NHIA (Hispanic, Non-Hisp): Non-Spanish-Hispanic-Latino
                186: Origin recode NHIA (Hispanic, Non-Hisp): Spanish-Hispanic-Latino
                187: Grade Recode (thru 2017): 1
                188: Grade Recode (thru 2017): 2
                189: Grade Recode (thru 2017): 3
                190: Grade Recode (thru 2017): 4
                191: Grade Recode (thru 2017): 9
                192: Age_group
                193: Year of diagnosis_std
                194: Total number of in situ/malignant tumors for patient_std
                195: Total number of benign/borderline tumors for patient_std
                196: CS tumor size (2004-2015)_std
                197: Regional nodes examined (1988+)_std
                198: Regional nodes positive (1988+)_std

            """        
            new_features = np.zeros((features.shape[0], 199))
    
            # set the `categories` to avoid missing any categories that did not show up in the training set
            ohe_encoder = OneHotEncoder(categories=[[i for i in range(len(cat))] for cat in ord_enc.categories_])
            num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            scaler = StandardScaler()
            
            # categorical: OHE
            new_features[:, :192] = ohe_encoder.fit_transform(features[:, :15].astype('int32')).toarray()
            # age group: leave as it is
            new_features[:, 192] = features[:, 15]
            # Year of diagnosis + numerical_col: standardize
            new_features[:, 193:] = scaler.fit_transform(num_imputer.fit_transform(features[:, 16:]).astype(float))
            
            if cox:
                remove_cols = [4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 19, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 45, 48, 49, 52, 56, 57, 58, 59, 61, 63, 64, 67, 71, 75, 81, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 143, 147, 150, 151, 153, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 180, 183, 185, 186, 188, 190, 193]
                # if cox == 'csCox':
                #     remove_cols.extend([0, 1, 2, 3, 5, 73, 76, 77, 82])
                new_features = np.delete(new_features, remove_cols, axis=1)
                
            return new_features, (ord_enc, ohe_encoder, num_imputer, scaler)

        def apply_preprocessor(features, preprocessor, cox=False):
            new_features = np.zeros((features.shape[0], 199))
            ord_enc, ohe_encoder, num_imputer, scaler = preprocessor

            new_features[:, :192] = ohe_encoder.transform(features[:, :15].astype('int32')).toarray()
            new_features[:, 192] = features[:, 15]
            new_features[:, 193:] = scaler.transform(num_imputer.transform(features[:, 16:]).astype(float))
            if cox:
                remove_cols = [4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 19, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 45, 48, 49, 52, 56, 57, 58, 59, 61, 63, 64, 67, 71, 75, 81, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 143, 147, 150, 151, 153, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 180, 183, 185, 186, 188, 190, 193]
                # if cox == 'csCox':
                #     remove_cols.extend([0, 1, 2, 3, 5, 73, 76, 77, 82])
                new_features = np.delete(new_features, remove_cols, axis=1)
            return new_features
        
        dataset_random_seed = 335585357
        train_test_split_prespecified = False


    elif dataset == 'seer':
        if not competing:
            raise ValueError(f'For seer dataset, we currently only support competing=True, but got competing={competing}.')
        if time_series:
            raise ValueError(f'For seer dataset, we currently only support time_series=False, but got time_series={time_series}.')
        
        # The following code is cited from the below link with minor changes:
        # https://github.com/Jeanselme/NeuralFineGray/blob/main/nfg/datasets.py#L39
        df = pd.read_csv('data/seer_test (trying to match vincent, with selection, fixed grade, nov 2024 sub).csv')
        # Remove multiple visits
        df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008'])

        # Grade Recode (thru 2017)
        df["Grade Recode (thru 2017)"].replace('Well differentiated; Grade I', '1', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Moderately differentiated; Grade II', '2', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Poorly differentiated; Grade III', '3', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Undifferentiated; anaplastic; Grade IV', '4', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Unknown', '9', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Blank(s)', 'NaN', inplace = True)

        # Grade Recode (2018+) Added by anonymous authors to deal with the 2018+ change in how grade is coded
        df["Derived Summary Grade 2018 (2018+)"].replace('A', '1', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('B', '2', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('C', '3', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('D', '4', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('L', '1', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('M', '2', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('H', '3', inplace = True)
        df["Derived Summary Grade 2018 (2018+)"].replace('Unknown', '9', inplace = True)
        df["Grade Recode (thru 2017)"].replace('Blank(s)', 'NaN', inplace = True)

        # Create unified grade column: 1, 2, 3, 4; 9 for unknown;
        # leave Unknown as it is instead of treating it as "missing". 
        # This means we will not try to impute Unknown entries for Grade
        df["Grade_unified"] = ''
        df.loc[df['Year of diagnosis']<2018, 'Grade_unified'] = df.loc[df['Year of diagnosis']<2018, 'Grade Recode (thru 2017)']
        df.loc[df['Year of diagnosis']>=2018, 'Grade_unified'] = df.loc[df['Year of diagnosis']>=2018, 'Derived Summary Grade 2018 (2018+)']
        df = df.drop(columns=['Grade Recode (thru 2017)', 'Derived Summary Grade 2018 (2018+)'])

        # Encode using dictionary to remove missing data
        df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)
        df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
        df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
        df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
        df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

        # Remove patients without survival time
        df = df[~df.duration.isna()]

        # Outcome
        df['duration'] = df['duration'].astype(float)
        df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)" # Death
        df.loc[(df["COD to site recode"] == "Diseases of Heart") & \
               (df["SEER cause-specific death classification"] == "Alive or dead of other cause"), 'event'] = 2 # CVD

        df = df.drop(columns = ["COD to site recode"])
        df['between_2004_2015'] = df['Year of diagnosis'].between(2004, 2015).astype(int) # Modified by anonymous authors: add indicator for changes in Collaborative Stage system

        ## Categorical (Modified by Xanonymous authors):
        #   ["Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant"] are duplicated features when we only look at malignant cancer cohort.
        #   Both of them are removed as they are too granular,  while SEER dataset provides "Histology recode - broad groupings", 
        #   which map Histologic Type into broader categories for easier analysis.
        binary_col = ["between_2004_2015"]
        categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality",
                "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
                "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
                "Sequence number", "RX Summ--Surg Prim Site (1998+)",
                "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)",
                "Grade_unified"]
        ordinal_col = ["Age recode with <1 year olds", "Year of diagnosis"]
        ## Numerical
        numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
            "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
        

        df[ordinal_col] = df[ordinal_col].replace(
        {age: number
            for number, age in enumerate(['01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
            '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years',
            '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
        }).replace({
            grade: number
            for number, grade in enumerate(['1', '2', '3', '4'])
        })

        cat_imputer = SimpleImputer(strategy='most_frequent')
        ord_enc = OrdinalEncoder()
        df[categorical_col] = ord_enc.fit_transform(cat_imputer.fit_transform(df[categorical_col]).astype(str))
        df[ordinal_col] = cat_imputer.fit_transform(df[ordinal_col])
        
        number_of_cat_per_cat_col = dict(df[categorical_col].nunique())
        cat_features_after_preprocessing = [f'{col}: {ord_enc.categories_[col_idx][cat_idx]}' for col_idx, col in enumerate(categorical_col) for cat_idx in range(number_of_cat_per_cat_col[col])]
        
        features_before_preprocessing = binary_col + categorical_col + ordinal_col + numerical_col
        features_after_preprocessing = binary_col + \
            cat_features_after_preprocessing + \
                ["Age_group", "Year of diagnosis_std"] + \
                [f"{col}_std" for col in numerical_col]
        
        features = df[features_before_preprocessing].to_numpy()
        observed_times = df['duration'].to_numpy().astype('float32')
        event_indicators = df['event'].to_numpy().astype('float32')
        events = ['BC', 'CVD']

        X = features
        Y = observed_times
        D = event_indicators

        def build_preprocessor_and_preprocess(features):
            """
            Prior to preprocessing, the features are expected to be:
                0: between_2004_2015
                1: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)
                2: Laterality
                3: Diagnostic Confirmation
                4: Histology recode - broad groupings
                5: Chemotherapy recode (yes, no/unk)
                6: Radiation recode
                7: ER Status Recode Breast Cancer (1990+)
                8: PR Status Recode Breast Cancer (1990+)
                9: Sequence number
                10: RX Summ--Surg Prim Site (1998+)
                11: CS extension (2004-2015)
                12: CS lymph nodes (2004-2015)
                13: CS mets at dx (2004-2015)
                14: Origin recode NHIA (Hispanic, Non-Hisp)
                15: Grade_unified
                16: Age recode with <1 year olds
                17: Year of diagnosis
                18: Total number of in situ/malignant tumors for patient
                19: Total number of benign/borderline tumors for patient
                20: CS tumor size (2004-2015)
                21: Regional nodes examined (1988+)
                22: Regional nodes positive (1988+)

            After preprocessing, the new features are:
                0: between_2004_2015
                1: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Hispanic (All Races)
                2: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic American Indian/Alaska Native
                3: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Asian or Pacific Islander
                4: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Black
                5: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic Unknown Race
                6: Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic): Non-Hispanic White
                7: Laterality: Bilateral, single primary
                8: Laterality: Left - origin of primary
                9: Laterality: Only one side - side unspecified
                10: Laterality: Paired site, but no information concerning laterality
                11: Laterality: Right - origin of primary
                12: Diagnostic Confirmation: Clinical diagnosis only
                13: Diagnostic Confirmation: Direct visualization without microscopic confirmation
                14: Diagnostic Confirmation: Positive exfoliative cytology, no positive histology
                15: Diagnostic Confirmation: Positive histology
                16: Diagnostic Confirmation: Positive laboratory test/marker study
                17: Diagnostic Confirmation: Positive microscopic confirm, method not specified
                18: Diagnostic Confirmation: Radiography without microscopic confirm
                19: Histology recode - broad groupings: 8000-8009: unspecified neoplasms
                20: Histology recode - broad groupings: 8010-8049: epithelial neoplasms, NOS
                21: Histology recode - broad groupings: 8050-8089: squamous cell neoplasms
                22: Histology recode - broad groupings: 8090-8119: basal cell neoplasms
                23: Histology recode - broad groupings: 8120-8139: transitional cell papillomas and carcinomas
                24: Histology recode - broad groupings: 8140-8389: adenomas and adenocarcinomas
                25: Histology recode - broad groupings: 8390-8429: adnexal and skin appendage neoplasms
                26: Histology recode - broad groupings: 8430-8439: mucoepidermoid neoplasms
                27: Histology recode - broad groupings: 8440-8499: cystic, mucinous and serous neoplasms
                28: Histology recode - broad groupings: 8500-8549: ductal and lobular neoplasms
                29: Histology recode - broad groupings: 8550-8559: acinar cell neoplasms
                30: Histology recode - broad groupings: 8560-8579: complex epithelial neoplasms
                31: Histology recode - broad groupings: 8680-8719: paragangliomas and glumus tumors
                32: Histology recode - broad groupings: 8720-8799: nevi and melanomas
                33: Histology recode - broad groupings: 8800-8809: soft tissue tumors and sarcomas, NOS
                34: Histology recode - broad groupings: 8810-8839: fibromatous neoplasms
                35: Histology recode - broad groupings: 8840-8849: myxomatous neoplasms
                36: Histology recode - broad groupings: 8850-8889: lipomatous neoplasms
                37: Histology recode - broad groupings: 8890-8929: myomatous neoplasms
                38: Histology recode - broad groupings: 8930-8999: complex mixed and stromal neoplasms
                39: Histology recode - broad groupings: 9000-9039: fibroepithelial neoplasms
                40: Histology recode - broad groupings: 9040-9049: synovial-like neoplasms
                41: Histology recode - broad groupings: 9120-9169: blood vessel tumors
                42: Histology recode - broad groupings: 9180-9249: osseous and chondromatous neoplasms
                43: Histology recode - broad groupings: 9260-9269: miscellaneous bone tumors (C40._, C41._)
                44: Histology recode - broad groupings: 9350-9379: miscellaneous tumors
                45: Histology recode - broad groupings: 9380-9489: gliomas
                46: Histology recode - broad groupings: 9540-9579: nerve sheath tumors
                47: Histology recode - broad groupings: 9580-9589: granular cell tumors & alveolar soft part sarcoma
                48: Chemotherapy recode (yes, no/unk): No/Unknown
                49: Chemotherapy recode (yes, no/unk): Yes
                50: Radiation recode: Beam radiation
                51: Radiation recode: Combination of beam with implants or isotopes
                52: Radiation recode: None/Unknown
                53: Radiation recode: Radiation, NOS  method or source not specified
                54: Radiation recode: Radioactive implants (includes brachytherapy) (1988+)
                55: Radiation recode: Radioisotopes (1988+)
                56: Radiation recode: Recommended, unknown if administered
                57: Radiation recode: Refused (1988+)
                58: ER Status Recode Breast Cancer (1990+): Borderline/Unknown
                59: ER Status Recode Breast Cancer (1990+): Negative
                60: ER Status Recode Breast Cancer (1990+): Positive
                61: ER Status Recode Breast Cancer (1990+): Recode not available
                62: PR Status Recode Breast Cancer (1990+): Borderline/Unknown
                63: PR Status Recode Breast Cancer (1990+): Negative
                64: PR Status Recode Breast Cancer (1990+): Positive
                65: PR Status Recode Breast Cancer (1990+): Recode not available
                66: Sequence number: 10th of 10 or more primaries
                67: Sequence number: 11th of 11 or more primaries
                68: Sequence number: 13th of 13 or more primaries
                69: Sequence number: 1st of 2 or more primaries
                70: Sequence number: 2nd of 2 or more primaries
                71: Sequence number: 3rd of 3 or more primaries
                72: Sequence number: 4th of 4 or more primaries
                73: Sequence number: 5th of 5 or more primaries
                74: Sequence number: 6th of 6 or more primaries
                75: Sequence number: 7th of 7 or more primaries
                76: Sequence number: 8th of 8 or more primaries
                77: Sequence number: One primary only
                78: RX Summ--Surg Prim Site (1998+): 00
                79: RX Summ--Surg Prim Site (1998+): 19
                80: RX Summ--Surg Prim Site (1998+): 20
                81: RX Summ--Surg Prim Site (1998+): 21
                82: RX Summ--Surg Prim Site (1998+): 22
                83: RX Summ--Surg Prim Site (1998+): 23
                84: RX Summ--Surg Prim Site (1998+): 24
                85: RX Summ--Surg Prim Site (1998+): 30
                86: RX Summ--Surg Prim Site (1998+): 40
                87: RX Summ--Surg Prim Site (1998+): 41
                88: RX Summ--Surg Prim Site (1998+): 42
                89: RX Summ--Surg Prim Site (1998+): 43
                90: RX Summ--Surg Prim Site (1998+): 44
                91: RX Summ--Surg Prim Site (1998+): 45
                92: RX Summ--Surg Prim Site (1998+): 46
                93: RX Summ--Surg Prim Site (1998+): 47
                94: RX Summ--Surg Prim Site (1998+): 48
                95: RX Summ--Surg Prim Site (1998+): 49
                96: RX Summ--Surg Prim Site (1998+): 50
                97: RX Summ--Surg Prim Site (1998+): 51
                98: RX Summ--Surg Prim Site (1998+): 52
                99: RX Summ--Surg Prim Site (1998+): 53
                100: RX Summ--Surg Prim Site (1998+): 54
                101: RX Summ--Surg Prim Site (1998+): 55
                102: RX Summ--Surg Prim Site (1998+): 56
                103: RX Summ--Surg Prim Site (1998+): 57
                104: RX Summ--Surg Prim Site (1998+): 58
                105: RX Summ--Surg Prim Site (1998+): 59
                106: RX Summ--Surg Prim Site (1998+): 60
                107: RX Summ--Surg Prim Site (1998+): 61
                108: RX Summ--Surg Prim Site (1998+): 62
                109: RX Summ--Surg Prim Site (1998+): 63
                110: RX Summ--Surg Prim Site (1998+): 64
                111: RX Summ--Surg Prim Site (1998+): 65
                112: RX Summ--Surg Prim Site (1998+): 66
                113: RX Summ--Surg Prim Site (1998+): 67
                114: RX Summ--Surg Prim Site (1998+): 68
                115: RX Summ--Surg Prim Site (1998+): 69
                116: RX Summ--Surg Prim Site (1998+): 70
                117: RX Summ--Surg Prim Site (1998+): 71
                118: RX Summ--Surg Prim Site (1998+): 72
                119: RX Summ--Surg Prim Site (1998+): 73
                120: RX Summ--Surg Prim Site (1998+): 74
                121: RX Summ--Surg Prim Site (1998+): 75
                122: RX Summ--Surg Prim Site (1998+): 76
                123: RX Summ--Surg Prim Site (1998+): 80
                124: RX Summ--Surg Prim Site (1998+): 90
                125: RX Summ--Surg Prim Site (1998+): 99
                126: CS extension (2004-2015): 050
                127: CS extension (2004-2015): 070
                128: CS extension (2004-2015): 100
                129: CS extension (2004-2015): 110
                130: CS extension (2004-2015): 120
                131: CS extension (2004-2015): 130
                132: CS extension (2004-2015): 140
                133: CS extension (2004-2015): 170
                134: CS extension (2004-2015): 180
                135: CS extension (2004-2015): 190
                136: CS extension (2004-2015): 200
                137: CS extension (2004-2015): 300
                138: CS extension (2004-2015): 400
                139: CS extension (2004-2015): 410
                140: CS extension (2004-2015): 510
                141: CS extension (2004-2015): 512
                142: CS extension (2004-2015): 514
                143: CS extension (2004-2015): 516
                144: CS extension (2004-2015): 518
                145: CS extension (2004-2015): 519
                146: CS extension (2004-2015): 520
                147: CS extension (2004-2015): 575
                148: CS extension (2004-2015): 580
                149: CS extension (2004-2015): 585
                150: CS extension (2004-2015): 600
                151: CS extension (2004-2015): 605
                152: CS extension (2004-2015): 610
                153: CS extension (2004-2015): 612
                154: CS extension (2004-2015): 613
                155: CS extension (2004-2015): 615
                156: CS extension (2004-2015): 620
                157: CS extension (2004-2015): 680
                158: CS extension (2004-2015): 710
                159: CS extension (2004-2015): 715
                160: CS extension (2004-2015): 725
                161: CS extension (2004-2015): 730
                162: CS extension (2004-2015): 750
                163: CS extension (2004-2015): 780
                164: CS extension (2004-2015): 790
                165: CS extension (2004-2015): 950
                166: CS extension (2004-2015): 999
                167: CS lymph nodes (2004-2015): 000
                168: CS lymph nodes (2004-2015): 050
                169: CS lymph nodes (2004-2015): 130
                170: CS lymph nodes (2004-2015): 150
                171: CS lymph nodes (2004-2015): 155
                172: CS lymph nodes (2004-2015): 250
                173: CS lymph nodes (2004-2015): 255
                174: CS lymph nodes (2004-2015): 257
                175: CS lymph nodes (2004-2015): 258
                176: CS lymph nodes (2004-2015): 260
                177: CS lymph nodes (2004-2015): 280
                178: CS lymph nodes (2004-2015): 500
                179: CS lymph nodes (2004-2015): 510
                180: CS lymph nodes (2004-2015): 520
                181: CS lymph nodes (2004-2015): 600
                182: CS lymph nodes (2004-2015): 610
                183: CS lymph nodes (2004-2015): 620
                184: CS lymph nodes (2004-2015): 630
                185: CS lymph nodes (2004-2015): 710
                186: CS lymph nodes (2004-2015): 720
                187: CS lymph nodes (2004-2015): 730
                188: CS lymph nodes (2004-2015): 735
                189: CS lymph nodes (2004-2015): 740
                190: CS lymph nodes (2004-2015): 745
                191: CS lymph nodes (2004-2015): 748
                192: CS lymph nodes (2004-2015): 750
                193: CS lymph nodes (2004-2015): 755
                194: CS lymph nodes (2004-2015): 760
                195: CS lymph nodes (2004-2015): 763
                196: CS lymph nodes (2004-2015): 764
                197: CS lymph nodes (2004-2015): 765
                198: CS lymph nodes (2004-2015): 768
                199: CS lymph nodes (2004-2015): 770
                200: CS lymph nodes (2004-2015): 780
                201: CS lymph nodes (2004-2015): 800
                202: CS lymph nodes (2004-2015): 805
                203: CS lymph nodes (2004-2015): 810
                204: CS lymph nodes (2004-2015): 815
                205: CS lymph nodes (2004-2015): 820
                206: CS lymph nodes (2004-2015): 999
                207: CS mets at dx (2004-2015): 00
                208: CS mets at dx (2004-2015): 05
                209: CS mets at dx (2004-2015): 07
                210: CS mets at dx (2004-2015): 10
                211: CS mets at dx (2004-2015): 40
                212: CS mets at dx (2004-2015): 42
                213: CS mets at dx (2004-2015): 44
                214: CS mets at dx (2004-2015): 50
                215: CS mets at dx (2004-2015): 60
                216: CS mets at dx (2004-2015): 99
                217: Origin recode NHIA (Hispanic, Non-Hisp): Non-Spanish-Hispanic-Latino
                218: Origin recode NHIA (Hispanic, Non-Hisp): Spanish-Hispanic-Latino
                219: Grade_unified: 1
                220: Grade_unified: 2
                221: Grade_unified: 3
                222: Grade_unified: 4
                223: Grade_unified: 9
                224: Age_group
                225: Year of diagnosis_std
                226: Total number of in situ/malignant tumors for patient_std
                227: Total number of benign/borderline tumors for patient_std
                228: CS tumor size (2004-2015)_std
                229: Regional nodes examined (1988+)_std
                230: Regional nodes positive (1988+)_std
            """        
            new_features = np.zeros((features.shape[0], 231))
    
            ohe_encoder = OneHotEncoder(categories=[[i for i in range(len(cat))] for cat in ord_enc.categories_])
            num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            scaler = StandardScaler()
            
            # binary
            new_features[:, 0] = features[:, 0]
            # categorical: OHE
            new_features[:, 1:224] = ohe_encoder.fit_transform(features[:, 1:16].astype('int32')).toarray()
            # age group: leave as it is
            new_features[:, 224] = features[:, 16]
            # Year of diagnosis + numerical_col: standardize
            new_features[:, 225:] = scaler.fit_transform(num_imputer.fit_transform(features[:, 17:]).astype(float))
            
            return new_features, (ord_enc, ohe_encoder, num_imputer, scaler)

        def apply_preprocessor(features, preprocessor):
            new_features = np.zeros((features.shape[0], 231))
            ord_enc, ohe_encoder, num_imputer, scaler = preprocessor

            new_features[:, 0] = features[:, 0]
            new_features[:, 1:224] = ohe_encoder.transform(features[:, 1:16].astype('int32')).toarray()
            new_features[:, 224] = features[:, 16]
            new_features[:, 225:] = scaler.transform(num_imputer.transform(features[:, 17:]).astype(float))
            return new_features
        
        dataset_random_seed = 335585357
        train_test_split_prespecified = False

    
    elif dataset == 'synthetic':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0

        features_before_preprocessing = df.drop(columns=['event', 'duration']).columns.tolist()
        features_after_preprocessing = [f'{col}_std' for col in features_before_preprocessing]

        features = df[features_before_preprocessing].values.astype('float32')
        observed_times = df['duration'].values.astype('float32')
        event_indicators = df['event'].values.astype('int32')
        events = ['Event1', 'Event2']

        X = features
        Y = observed_times
        D = event_indicators

        def build_preprocessor_and_preprocess(features, cox=False):
            scaler = StandardScaler()
            new_features = scaler.fit_transform(features)
            return new_features, scaler
        
        def apply_preprocessor(features, preprocessor, cox=False):
            scaler = preprocessor
            new_features = scaler.transform(features)
            return new_features
        
        dataset_random_seed = 674540762
        train_test_split_prespecified = False

    
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    if train_test_split_prespecified:
        # this first case corresponds to when the dataset has pre-specified
        # training and test data (so that the variables `X_train`, `y_train`,
        # `X_test`, and `y_test` have been defined earlier already)

        # shuffle only the training data (do not modify `X_test` and `y_test`
        # that are defined earlier)
        rng = np.random.RandomState(dataset_random_seed + random_seed_offset)
        shuffled_indices = rng.permutation(len(X_train))

        if type(X_train) == np.ndarray:
            X_train = X_train[shuffled_indices]
            Y_train = Y_train[shuffled_indices]
            D_train = D_train[shuffled_indices]
        else:
            assert type(X_train) == list
            X_train = [X_train[idx] for idx in shuffled_indices]
            Y_train = [Y_train[idx] for idx in shuffled_indices]
            D_train = [D_train[idx] for idx in shuffled_indices]
    else:
        # this second case corresponds to when the dataset does *not* have
        # pre-specified training and test data, so we need to define
        # `X_train`, `y_train`, `X_test`, and `y_test` first via a random split
        # (we use sklearn's `train_test_split` function)

        # note that by default, sklearn's `train_test_split` will shuffle the
        # data before doing the split
        if fix_test_shuffle_train:
            # first come up with training and test data
            rng = np.random.RandomState(dataset_random_seed)
            X_train, X_test, Y_train, Y_test, D_train, D_test = \
                train_test_split(X, Y, D, test_size=test_size, random_state=rng)

            if random_seed_offset > 0:
                # do another shuffle of only the training data using the seed
                # offset (at this point, we treat the test set defined by
                # `X_test` and `y_test` as fixed)
                rng = np.random.RandomState(dataset_random_seed
                                            + random_seed_offset)
                shuffled_indices = rng.permutation(len(X_train))

                if type(X_train) == np.ndarray:
                    X_train = X_train[shuffled_indices]
                    Y_train = Y_train[shuffled_indices]
                    D_train = D_train[shuffled_indices]
                else:
                    assert type(X_train) == list
                    X_train = [X_train[idx] for idx in shuffled_indices]
                    Y_train = [Y_train[idx] for idx in shuffled_indices]
                    D_train = [D_train[idx] for idx in shuffled_indices]
        else:
            # in this case, we do not treat the test data as fixed so that
            # different `random_seed_offset` values will yield different
            # train/test splits
            rng = np.random.RandomState(dataset_random_seed
                                        + random_seed_offset)
            X_train, X_test, Y_train, Y_test, D_train, D_test = \
                train_test_split(X, Y, D, test_size=test_size, random_state=rng)

    if diff_train_size:
        train_subset_size = int(len(X_train) * train_data_subset_rate)
        X_train = X_train[:train_subset_size]
        Y_train = Y_train[:train_subset_size]
        D_train = D_train[:train_subset_size]

    return X_train, Y_train, D_train, X_test, Y_test, D_test, \
            features_before_preprocessing, features_after_preprocessing, \
            events, train_test_split_prespecified, \
            build_preprocessor_and_preprocess, apply_preprocessor



class LabTransformCR(LabTransDiscreteTime):
    '''
    Label transformer for competing risks
    From the pycox demo of deephit competing risks:
        https://github.com/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb
    '''
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')