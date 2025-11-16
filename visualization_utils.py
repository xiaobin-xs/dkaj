import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import pickle

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.impute import SimpleImputer
plt.style.use('seaborn-v0_8-muted')

def heatmap_plot(heatmap, cif_max_time, feature_names,
                 all_n_bins_to_use, max_features_to_display=None,
                 max_observed_times=None, save_filename=None, show_plot=True,
                 units=None, custom_xlabel=None, axhline_xmin=0,
                 cluster_sizes=None, cluster_size_threshold=None):
    feature_names = np.array(feature_names).copy()
    heatmap = heatmap.copy()

    # sort by max probability first (and then we do another sort)
    if cluster_size_threshold is not None:
        cluster_mask = (np.array(cluster_sizes) >= cluster_size_threshold)
    else:
        cluster_mask = None
    new_order, new_all_n_bins_to_use \
        = argsort_feature_names(feature_names, heatmap, all_n_bins_to_use,
                                np.max, max, reverse=True,
                                cluster_mask=cluster_mask)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    # sort by peak-to-peak
    new_order, new_all_n_bins_to_use \
        = argsort_feature_names(feature_names, heatmap, new_all_n_bins_to_use,
                                np.ptp, max, reverse=True,
                                max_features_to_display=max_features_to_display,
                                cluster_mask=cluster_mask)
    feature_names = feature_names[new_order]
    heatmap = heatmap[new_order]

    start_indices = []
    start_idx = 0
    for n_bins_to_use in new_all_n_bins_to_use:
        start_indices.append(start_idx)
        start_idx += n_bins_to_use

    for idx, f in enumerate(feature_names):
        if 'bin#1(-inf' in f:
            idx2 = f.index('bin#1(-inf')
            feature_names[idx] = f[:idx2] + '< ' + f[idx2+11:-1]
        elif 'bin#' in f and 'inf)' in f:
            idx2 = f.index('bin#')
            feature_names[idx] = f[:idx2] + '≥ ' + f[idx2+6:-1].split(',')[0]
        elif 'bin#' in f:
            idx2 = f.index('bin#')
            feature_names[idx] = f.replace(f[idx2:idx2+5], 'in ')
        elif 'cat#' in f:
            idx2 = f.index('cat#')
            idx3 = f[idx2+4:].index('(')
            feature_names[idx] = f[:idx2] + '= ' + f[idx2+5+idx3:-1]

    with plt.style.context('seaborn-dark'):
        fig = plt.figure(figsize=(len(cif_max_time)*1.0,
                                  len(feature_names)/3*0.7),
                         dpi=300)
        ax = fig.gca()
        ax.xaxis.tick_top()
        ax.set_xlabel('X LABEL')
        ax.xaxis.set_label_position('top')
        # sns.heatmap(heatmap, cmap=modified_red_color_map)
        sns.heatmap(heatmap, cmap=modified_gray_color_map)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ticks = list(cbar.get_ticks())
        heatmap_max = heatmap.max()
        for tick_idx, tick in enumerate(ticks):
            if tick >= heatmap_max:
                ticks = ticks[:tick_idx]
                break
        cbar.set_ticks(ticks + [heatmap.max()])

        for start_idx in start_indices[1:]:
            plt.axhline(y=start_idx, color='black', linestyle='dashed',
                        linewidth=.5, xmin=axhline_xmin, xmax=1, clip_on=False)

        plt.ylabel('Feature')
        if custom_xlabel is None:
            if units is None:
                xlabel = 'Clusters sorted by median survival time'
            else:
                xlabel = 'Clusters sorted by median survival time (%s)' % units
            if cluster_sizes is not None:
                xlabel += '; cluster sizes are stated in square brackets'
            plt.xlabel(xlabel)
        else:
            plt.xlabel(custom_xlabel)
        plt.yticks(np.arange(len(feature_names)) + 0.5,
                   list(feature_names),
                   rotation='horizontal')
        if max_observed_times is not None:
            if cluster_sizes is None:
                plt.xticks(np.arange(len(cif_max_time)) + 0.5,
                           [np.isinf(median_time) and (" >%.2f" % max_time)
                            or " %.2f " % median_time
                            for median_time, max_time
                            in zip(cif_max_time, max_observed_times)])
            else:
                plt.xticks(np.arange(len(cif_max_time)) + 0.5,
                           [(np.isinf(median_time) and (" >%.2f" % max_time)
                             or " %.2f " % median_time)
                            + "\n[%d]" % cluster_size
                            for median_time, max_time, cluster_size
                            in zip(cif_max_time,
                                   max_observed_times,
                                   cluster_sizes)])
        if save_filename is not None:
            plt.savefig(save_filename, bbox_inches='tight')
        if not show_plot:
            plt.close()


def transform(dataset, X, continuous_n_bins=5):
    if dataset == 'pbc':
        feature_names = [
            'D-penicil',                                    # 0
            'female', 'ascites', 'hepatomegaly', 'spiders', # 1,2,3,4
            'edema','histologic','serBilir','serChol',      # 5,6,7,8
            'albumin','alkaline','SGOT','platelets',        # 9,10,11,12
            'prothrombin','age'                             # 13,14
        ]

        X_imputed = np.zeros_like(X)

        # binary features
        binary_indices = [0, 1, 2, 3, 4]          # 'D-penicil','female', 'ascites', 'hepatomegaly', 'spiders'
        # categorical
        categorical_indices = [5, 6]              # 'edema', 'histologic'
        # continuous
        continuous_indices = [7, 8, 9, 10, 11, 12, 13, 14]

        # 1. Impute binary features with most frequent (mode)
        imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_imputed[:, binary_indices] = imputer_mode.fit_transform(
            X[:, binary_indices]
        )

        # 2. Impute categorical features (also mode)
        for idx in categorical_indices:
            imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_imputed[:, idx:idx+1] = imputer_cat.fit_transform(
                X[:, idx:idx+1]
            )

        # 3. Impute continuous features with mean
        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed[:, continuous_indices] = imputer_mean.fit_transform(
            X[:, continuous_indices]
        )

        X = X_imputed

        leave_indices = [0, 1, 2, 3, 4]

        continuous_indices = [7, 8, 9, 10, 11, 12, 13, 14]
        discretized_features = []
        discretized_feature_names = []
        all_n_bins_to_use = []
        for idx in continuous_indices:
            n_bins_to_use = continuous_n_bins
            discretizer = KBinsDiscretizer(n_bins=n_bins_to_use,
                                           strategy='quantile',
                                           encode='onehot-dense')
            new_features = discretizer.fit_transform(
                X[:, idx].reshape(-1, 1).astype(float))
            if discretizer.n_bins_[0] != n_bins_to_use:
                n_bins_to_use = discretizer.n_bins_[0]
            if n_bins_to_use > 1:
                discretized_features.append(new_features)
                for bin_idx in range(n_bins_to_use):
                    if bin_idx == 0:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#1(-inf,%.2f)'
                            % discretizer.bin_edges_[0][bin_idx+1])
                    elif bin_idx == n_bins_to_use - 1:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,inf)'
                            % (n_bins_to_use,
                               discretizer.bin_edges_[0][bin_idx]))
                    else:
                        discretized_feature_names.append(
                            feature_names[idx]
                            + ' bin#%d[%.2f,%.2f)'
                            % tuple([bin_idx + 1] +
                                    list(discretizer.bin_edges_[0][
                                        bin_idx:bin_idx+2])))
            else:
                raise Exception('Single discretization bin encountered for '
                                + 'feature "%s"' % feature_names[idx])
            all_n_bins_to_use.append(n_bins_to_use)

        # handle categorical
        discretizer = OneHotEncoder(sparse=False)
        new_features = discretizer.fit_transform(X[:, [categorical_indices[0]]].astype(int))
        discretized_features.append(new_features)
        for cat_idx, cat in enumerate(discretizer.categories_[0]):
            discretized_feature_names.append(
                'edema %s' % (cat)
            )
        all_n_bins_to_use.append(len(discretizer.categories_[0]))

        discretizer = OneHotEncoder(sparse=False)
        new_features = discretizer.fit_transform(X[:, [categorical_indices[1]]].astype(int))
        discretized_features.append(new_features)
        for cat_idx, cat in enumerate(discretizer.categories_[0]):
            discretized_feature_names.append(
                'histologic stage %s' % (cat)
            )
        all_n_bins_to_use.append(len(discretizer.categories_[0]))
            

        for idx in leave_indices:
            discretized_features.append(X[:, idx].reshape(-1, 1).astype(float))
            discretized_feature_names.append(feature_names[idx])
            all_n_bins_to_use.append(1)

        return np.hstack(discretized_features), discretized_feature_names, \
            all_n_bins_to_use

    elif dataset == 'framingham':
        feature_names = [
            'FEMALE', 'CURSMOKE', 'DIABETES', 'BPMEDS', 
            'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 
            'educ', 
            'TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 
            'BMI', 'HEARTRTE', 'GLUCOSE'
        ]


        # indices
        binary_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]          # first 9
        categorical_indices = [9]                             # 'educ'
        continuous_indices = [10, 11, 12, 13, 14, 15, 16, 17] # last 8

        discretized_features = []
        discretized_feature_names = []
        all_n_bins_to_use = []

        # handle continuous variables with quantile binning
        for idx in continuous_indices:
            n_bins_to_use = continuous_n_bins
            discretizer = KBinsDiscretizer(
                n_bins=n_bins_to_use, strategy='quantile', encode='onehot-dense'
            )
            new_features = discretizer.fit_transform(
                X[:, idx].reshape(-1, 1).astype(float)
            )
            if discretizer.n_bins_[0] != n_bins_to_use:
                n_bins_to_use = discretizer.n_bins_[0]
            if n_bins_to_use > 1:
                discretized_features.append(new_features)
                for bin_idx in range(n_bins_to_use):
                    if bin_idx == 0:
                        discretized_feature_names.append(
                            feature_names[idx] + 
                            ' bin#1(-inf,%.2f)' % discretizer.bin_edges_[0][bin_idx+1]
                        )
                    elif bin_idx == n_bins_to_use - 1:
                        discretized_feature_names.append(
                            feature_names[idx] +
                            ' bin#%d[%.2f,inf)' % (n_bins_to_use,
                            discretizer.bin_edges_[0][bin_idx])
                        )
                    else:
                        discretized_feature_names.append(
                            feature_names[idx] +
                            ' bin#%d[%.2f,%.2f)' % tuple(
                                [bin_idx+1] + list(discretizer.bin_edges_[0][bin_idx:bin_idx+2])
                            )
                        )
            else:
                raise Exception(
                    'Single discretization bin encountered for feature "%s"' % feature_names[idx]
                )
            all_n_bins_to_use.append(n_bins_to_use)

        # handle binary variables (leave as-is)
        for idx in binary_indices:
            discretized_features.append(
                X[:, idx].reshape(-1, 1).astype(float)
            )
            discretized_feature_names.append(feature_names[idx])
            all_n_bins_to_use.append(1)

        # handle categorical 'educ'
        discretizer = OneHotEncoder(sparse=False)
        new_features = discretizer.fit_transform(X[:, [categorical_indices[0]]].astype(int))
        discretized_features.append(new_features)
        for cat_idx, cat in enumerate(discretizer.categories_[0]):
            discretized_feature_names.append(
                'educ cat#%d(%s)' % (cat_idx + 1, cat)
            )
        all_n_bins_to_use.append(len(discretizer.categories_[0]))

        return (
            np.hstack(discretized_features),
            discretized_feature_names,
            all_n_bins_to_use
        )

    elif dataset == 'seer2010':
        categorical_col = ["Race and origin (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality",
                "Diagnostic Confirmation", "Histology - broad groupings", "Chemotherapy (yes, no/unk)",
                "Radiation", "ER Status Breast Cancer", "PR Status Breast Cancer",
                "Sequence number", "RX Summ--Surg Prim Site",
                "CS extension", "CS lymph nodes", "CS mets at dx", 
                "Origin NHIA (Hispanic, Non-Hisp)", "Cancer Grade"] # size 15
        ordinal_col = ["Age group", "Year of diagnosis"]
        numerical_col = ["Total number of in situ/malignant tumors", "Total number of benign/borderline tumors",
            "CS tumor size", "Regional nodes examined", "Regional nodes positive"]
        feature_names = categorical_col + ordinal_col + numerical_col

        age_mapping = {
            age: number
            for number, age in enumerate([
                '01-04 years', '05-09 years', '10-14 years', '15-19 years',
                '20-24 years', '25-29 years', '30-34 years', '35-39 years',
                '40-44 years', '45-49 years', '50-54 years', '55-59 years',
                '60-64 years', '65-69 years', '70-74 years', '75-79 years',
                '80-84 years', '85+ years'
            ])
        }
        # Reverse dictionary
        reverse_age_mapping = {v: k for k, v in age_mapping.items()}
        age_decode_func = np.vectorize(lambda x: reverse_age_mapping[x])
        
        with open("visualization/seer2010_categories.pkl", "rb") as f:
            categories_loaded = pickle.load(f)

        def manual_inverse_transform(encoded, categories_loaded):
            decoded = []
            for row in encoded.astype(int):  # make sure ints
                decoded_row = [
                    categories_loaded[col_idx][val] for col_idx, val in enumerate(row)
                ]
                decoded.append(decoded_row)
            return np.array(decoded)

        X_cat = np.empty((X.shape[0], 16), dtype=object)
        X_cat[:, :len(categorical_col)] = manual_inverse_transform(X[:, :len(categorical_col)], categories_loaded)
        X_cat[:, -1] = age_decode_func(X[:, 15])

        # indices
        categorical_indices = list(range(len(categorical_col) + 1))
        continuous_indices = list(
            range(len(categorical_col+ordinal_col), len(feature_names))
            )

        discretized_features = []
        discretized_feature_names = []
        all_n_bins_to_use = []
        
        skipped_features = ["Race and origin (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Origin NHIA (Hispanic, Non-Hisp)",
                            'RX Summ--Surg Prim Site', 'CS lymph nodes',
                             'Diagnostic Confirmation', 'CS mets at dx', 'CS extension', 
                             'Histology - broad groupings']
        # handle continuous variables with quantile binning
        for idx in continuous_indices:
            if feature_names[idx] in skipped_features:
                continue
            if idx == 18:
                n_bins_to_use = 2
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins_to_use, strategy='uniform', encode='onehot-dense'
                )
            else:
                n_bins_to_use = continuous_n_bins
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins_to_use, strategy='quantile', encode='onehot-dense'
                )
            new_features = discretizer.fit_transform(
                X[:, idx].reshape(-1, 1).astype(float)
            )
            if discretizer.n_bins_[0] != n_bins_to_use:
                n_bins_to_use = discretizer.n_bins_[0]
            if n_bins_to_use > 1:
                discretized_features.append(new_features)
                for bin_idx in range(n_bins_to_use):
                    if bin_idx == 0:
                        discretized_feature_names.append(
                            feature_names[idx] + 
                            ' bin#1(-inf,%.2f)' % discretizer.bin_edges_[0][bin_idx+1]
                        )
                    elif bin_idx == n_bins_to_use - 1:
                        discretized_feature_names.append(
                            feature_names[idx] +
                            ' bin#%d[%.2f,inf)' % (n_bins_to_use,
                            discretizer.bin_edges_[0][bin_idx])
                        )
                    else:
                        discretized_feature_names.append(
                            feature_names[idx] +
                            ' bin#%d[%.2f,%.2f)' % tuple(
                                [bin_idx+1] + list(discretizer.bin_edges_[0][bin_idx:bin_idx+2])
                            )
                        )
            else:
                raise Exception(
                    'Single discretization bin encountered for feature "%s"' % feature_names[idx]
                )
            all_n_bins_to_use.append(n_bins_to_use)

        # handle categorical
        for idx in categorical_indices:
            feature_name = feature_names[idx]
            if feature_name in skipped_features:
                continue
            discretizer = OneHotEncoder(sparse=False)
            discretized_features.append(
                discretizer.fit_transform(X_cat[:, idx].reshape(-1, 1)))
            for cat_idx, cat in enumerate(discretizer.categories_[0]):
                discretized_feature_names.append(
                    feature_name + ' cat#%d(%s)' % (cat_idx + 1, cat.split('(')[0]))
            all_n_bins_to_use.append(len(discretizer.categories_[0]))

        return (
            np.hstack(discretized_features),
            discretized_feature_names,
            all_n_bins_to_use
        )
    
    else:
        raise NotImplementedError


def argsort_feature_names(feature_names, heatmap, all_n_bins_to_use,
                          score_func, aggregate_func, reverse=False,
                          max_features_to_display=None,
                          cluster_mask=None):
    scores = []
    start_indices = []
    start_idx = 0
    for n_bins_to_use in all_n_bins_to_use:
        start_indices.append(start_idx)
        overall_score = None
        for row_idx in range(start_idx, start_idx + n_bins_to_use):
            if cluster_mask is None:
                score = score_func(heatmap[row_idx])
            else:
                score = score_func(heatmap[:, cluster_mask][row_idx])
            if overall_score is None:
                overall_score = score
            else:
                score = aggregate_func(overall_score, score)
        scores.append(overall_score)
        start_idx += n_bins_to_use
    scores = np.array(scores)

    if reverse:
        sort_indices = np.argsort(-scores)
    else:
        sort_indices = np.argsort(scores)

    new_order = []
    new_all_n_bins_to_use = []
    n_displayed_features = 0
    for old_feature_idx in sort_indices:
        n_bins_to_use = all_n_bins_to_use[old_feature_idx]

        n_displayed_features += n_bins_to_use
        if max_features_to_display is not None and \
                n_displayed_features > max_features_to_display:
            break

        start_idx = start_indices[old_feature_idx]
        for idx in range(n_bins_to_use):
            new_order.append(start_idx + idx)
        new_all_n_bins_to_use.append(n_bins_to_use)

    return new_order, new_all_n_bins_to_use


colors = plt.cm.tab10.colors

modified_red_color_map = \
    mcolors.LinearSegmentedColormap(
        'modified_red_color_map',
        segmentdata={
            'red':   ((0.0,  1.0, 1.0),
                      (0.9,  0.9, 0.9),
                      (1.0,  0.6, 0.6)),
            'green': ((0.0,  1.0, 1.0),
                      (1.0,  0.0, 0.0)),
            'blue':  ((0.0,  1.0, 1.0),
                      (1.0,  0.0, 0.0))},
        N=256)


modified_gray_color_map = \
    mcolors.LinearSegmentedColormap(
        'modified_gray_color_map',
        segmentdata={
            'red':   ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.)),
            'green': ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.)),
            'blue':  ((0.0,  1.0, 1.0),
                      (0.2,  0.9, 0.9),
                      (1.0,  0., 0.))},
        N=256)
