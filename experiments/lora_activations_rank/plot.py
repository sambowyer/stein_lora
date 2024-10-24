import os
import pickle
import numpy as np

import matplotlib.pyplot as plt


# Directory containing the PCA objects
pca_dir = '/user/work/dg22309/stein_lora/experiments/lora_activations_rank/'
# Directory to save the plots
plots_dir = os.path.join(pca_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# dict to store cumulative variance numbers
cumulative_variance_dict = {}

# Iterate over all files in the PCA directory
for filename in os.listdir(pca_dir):
    if filename.endswith('.pkl'):
        # Load the PCA object
        with open(os.path.join(pca_dir, filename), 'rb') as file:
            pca = pickle.load(file)
        
        # # Plot the explained variance per principal component
        # plt.figure()
        # plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        # plt.title(f'Explained Variance per Principal Component\n{filename}')
        # plt.xlabel('Principal Component')
        # plt.ylabel('Explained Variance Ratio')
        # plt.grid(True)
        
        # # Save the plot
        # plot_filename = os.path.join(plots_dir, f'{os.path.splitext(filename)[0]}_explained_variance.png')
        # plt.savefig(plot_filename)
        # plt.close()

        r = filename.split('_')[1][1:]
        if len(filename.split('_')) > 2:
            seed = filename.split('_')[2][4:-4]
        else:
            seed = ""

        if r not in cumulative_variance_dict:
            cumulative_variance_dict[r] = []


        # Calculate the minimum number of components to explain 90%, 95%, and 99% of the variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
        num_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        num_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

        cumulative_variance_dict[r].append([num_components_80, num_components_90, num_components_95, num_components_99])


        # Plot the explained variance per principal component
        plt.figure()
        plt.plot(np.arange(1, 21), pca.explained_variance_ratio_[:20], marker='o')
        plt.title(f'Explained Variance per Principal Component (r={r}, seed={seed})\n80%: {num_components_80}, 90%: {num_components_90}, 95%: {num_components_95}, 99%: {num_components_99}')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        
        # Save the plot
        plot_filename = os.path.join(plots_dir, f'r{r}{"_" + str(seed) if seed != "" else ""}_explained_variance20.png')
        plt.savefig(plot_filename)
        plt.close()

# average the cumulative variance numbers over seeds and represent with numpy (by stacking)
cumulative_variance_dict_np = {r: np.stack(cumulative_variance_dict[r]).mean(0) for r in cumulative_variance_dict if ".pkl" not in r}
rs = [int(r) for r in list(cumulative_variance_dict_np.keys())]
rs.sort()
cumulative_variance_numbers = np.stack([cumulative_variance_dict_np[str(r)] for r in rs])

# Plot the number of components needed to explain 80%, 90%, 95%, and 99% of the variance for each r
plt.figure()
plt.plot(rs, cumulative_variance_numbers[:, 0], marker='o', label='80%')
plt.plot(rs, cumulative_variance_numbers[:, 1], marker='o', label='90%')
plt.plot(rs, cumulative_variance_numbers[:, 2], marker='o', label='95%')
plt.plot(rs, cumulative_variance_numbers[:, 3], marker='o', label='99%')
plt.title('Number of Components Needed to Explain Cumulative Variance\n(dimensions=768)')
plt.xlabel('LORA Rank r')
plt.xticks(rs)
plt.ylabel('Number of Components')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'cumulative_variance_numbers.png'))
plt.close()