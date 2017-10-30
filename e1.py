#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Some custom x-axis labelling to make our plots easier to read
def reduce_xaxis_labels(ax, factor):
    """Show only every ith label to prevent crowding on x-axis
        e.g. factor = 2 would plot every second x-axis label,
        starting at the first.

    Parameters
    ----------
    ax : matplotlib plot axis to be adjusted
    factor : int, factor to reduce the number of x-axis labels by
    """
    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    for label in ax.xaxis.get_ticklabels()[factor-1::factor]:
        label.set_visible(True)


if __name__ == '__main__':
    # Import TCGA melanoma data
    filename = 'data/counts.txt'
    with open(filename, 'rt') as f:
        data_table = pd.read_csv(f, index_col=0) # Parse file with pandas

    print(data_table.iloc[:5, :5])

    samples = list(data_table.columns)
    # Import gene lengths
    filename = 'data/genes.csv'
    with open(filename, 'rt') as f:
        # Parse file with pandas, index by GeneSymbol
        gene_info = pd.read_csv(f, index_col=0)


    #check shapes
    print("Genes in data_table: ", data_table.shape[0])
    print("Genes in gene_info: ", gene_info.shape[0])

    #lets only get the ones that exist in both dataframes
    # Subset gene info to match the count data
    matched_index = pd.Index.intersection(data_table.index, gene_info.index)

    # 2D ndarray containing expression counts for each gene in each individual
    counts = np.asarray(data_table.loc[matched_index], dtype=int)

    gene_names = np.array(matched_index)

    # 1D ndarray containing the lengths of each gene
    gene_lengths = np.asarray(gene_info.loc[matched_index]['GeneLength'], dtype=int)

    # Check how many genes and individuals were measured
    print(f'{counts.shape[0]} genes measured in {counts.shape[1]} individuals.')

    print(counts.shape)
    print(gene_lengths.shape)

    #commont to use
    #plt.style.use('ggplot')
    # Use our own style file for the plots so book matches
    plt.style.use('style/elegant.mplstyle')

    total_counts = np.sum(counts, axis=0)  # sum columns together
                                           # (axis=1 would sum rows)
    # Use Gaussian smoothing to estimate the density
    density = stats.kde.gaussian_kde(total_counts)

    # Make values for which to estimate the density, for plotting
    x = np.arange(min(total_counts), max(total_counts), 10000)

    # Make the density plot
    fig, ax = plt.subplots()
    ax.plot(x, density(x))
    ax.set_xlabel("Total counts per individual")
    ax.set_ylabel("Density")

    # plt.show()

    print(f'Count statistics:\n  min:  {np.min(total_counts)}'
          f'\n  mean: {np.mean(total_counts)}'
          f'\n  max:  {np.max(total_counts)}')

    # Subset data for plotting
    np.random.seed(seed=7) # Set seed so we will get consistent results
    # Randomly select 70 samples
    samples_index = np.random.choice(range(counts.shape[1]), size=70, replace=False)
    counts_subset = counts[:, samples_index]

    # Bar plot of expression counts by individual
    fig, ax = plt.subplots(figsize=(4.8, 2.4))

    with plt.style.context('style/thinner.mplstyle'):
        ax.boxplot(counts_subset)
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Gene expression counts")
        reduce_xaxis_labels(ax, 5)

    # plt.show()

    fig, ax = plt.subplots(figsize=(4.8,2.4))

    with plt.style.context('style/thinner.mplstyle'):
        ax.boxplot(np.log(counts_subset+1))
        ax.set_xlabel("Individuals")
        ax.set_ylabel('log gene expression counts')
        reduce_xaxis_labels(ax, 5)

    # plt.show()
    plt.clf()
    
    # Normalize by library size
    # Divide the expression counts by the total counts for that individual
    # Multiply by 1 million to get things back in a similar scale
    counts_lib_norm = counts / total_counts * 1000000
    # Notice how we just used broadcasting twice there!
    counts_subset_lib_norm = counts_lib_norm[:,samples_index]

    # Bar plot of expression counts by individual
    fig, ax = plt.subplots(figsize=(4.8, 2.4))

    with plt.style.context('style/thinner.mplstyle'):
        ax.boxplot(np.log(counts_subset_lib_norm + 1))
        ax.set_xlabel("Individuals")
        ax.set_ylabel("log gene expression counts")
        reduce_xaxis_labels(ax, 5)

    plt.show()
