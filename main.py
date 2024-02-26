import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict
from statistics import mean
from math import factorial

dist_mean = None
std_dev = None
distribution_prior = None
observations_per_sample = None
bin_map = []

@dataclass
class CutpointDistribution:
    dist_mean: float
    std_dev: float
    cut_points: list[list[float]]

def get_round_to_half(number):
    return round(number * 2) / 2

def get_partitions(num_samples:int, num_partitions:int, target_rating_rounded:float):

    stack = [[]]
    valid_partitions = []

    # Checking range avoids tolerace and double counting issues at boundaries
    target_rating_sum_inclusive_min = (target_rating_rounded - 0.25) * num_samples
    target_rating_sum_exclusive_max = (target_rating_rounded + 0.25) * num_samples

    while stack:

        partition = stack.pop()
        partition_rating_sum  = sum([(i+1)*partition[i] for i in range(len(partition))])

        #  If we have all partitions

        if len(partition)==num_partitions:
            if sum(partition) == num_samples and target_rating_sum_inclusive_min <= partition_rating_sum < target_rating_sum_exclusive_max:
                valid_partitions.append(partition)
            continue

        # If we have less than num_partitions

        for i in range(0, num_samples+1):
            # Ensure sample size is not exceeded
            # Ensure max rating is not excceded
            if sum(partition) + i <= num_samples and partition_rating_sum + i < target_rating_sum_exclusive_max: 
                stack.append(partition + [i])
            
    return valid_partitions

def get_partition_permutations_count(partition):
    
    # Calculate the denominator as the product of factorials of object counts
    denominator = 1
    for count in partition:
        denominator *= factorial(count)
    
    # Calculate the total number of permutations
    total_perms = factorial(sum(partition)) / denominator
    
    return total_perms

def get_partition_probability(partition, distribution_prior):

    iid_probability = 1.0
    
    if sum(partition)==0:
        return 0

    for i in range(len(partition)):

        lower_limit = distribution_prior.cut_points[i][0]
        upper_limit = distribution_prior.cut_points[i][1]
        event_probability = norm.cdf(upper_limit, loc=distribution_prior.dist_mean, scale=distribution_prior.std_dev) - norm.cdf(lower_limit, loc=distribution_prior.dist_mean, scale=distribution_prior.std_dev)
        iid_probability *= (event_probability**partition[i])
            
    return iid_probability


# Set Seaborn style
sns.set_theme(style="white")

# Title of the app
st.title("Conditional distribution of Q\u0303 given Q\u2217 and N_jt for the provider")

# Using sidebar for sliders with finer increments
with st.sidebar:
    dist_mean = st.slider(" Q\u2217 true mean service quality of the provider", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    std_dev = st.slider("Standard Deviation", min_value=1.0, max_value=5.0, value=3.3, step=0.1)

    q_1 = st.slider("Q1 1 star threshold (skyblue region)", min_value=-5.0, max_value=5.0, value=-3.0, step=0.1)
    q_2 = st.slider("Q2 2 star threshold (lightgreen region)", min_value=-5.0, max_value=5.0, value=-1.0, step=0.1)
    q_3 = st.slider("Q3 3 star threshold (salmon region)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    q_4 = st.slider("Q4 4 star threshold (wheat region)", min_value=-5.0, max_value=5.0, value=3.0, step=0.1)

    observations_per_sample = st.slider("Number of obervations per sample", min_value=1, max_value=25, value=10, step=1)

    distribution_prior = CutpointDistribution(dist_mean=dist_mean, 
                                              std_dev=std_dev, 
                                              cut_points=[[float("-inf"), q_1], [q_1, q_2], [q_2, q_3], [q_3, q_4], [q_4, float("inf")]])

# Ensure q1 < q2 < q3 < q4
if q_1 < q_2 < q_3 < q_4:

    # probability_observed_rating = 0.0

    for observed_rating in [x/10 for x in range(10, 51, 5)]:

        probability_observed_rating = 0.0

        partitions = get_partitions(num_samples=observations_per_sample, num_partitions=5, target_rating_rounded=observed_rating)

        for partition in partitions:
            permuation_count = get_partition_permutations_count(partition=partition)
            probability_partition = get_partition_probability(partition=partition, distribution_prior=distribution_prior)
            # probability_cumulative += permuation_count*probability_partition
            probability_observed_rating += permuation_count*probability_partition

        bin_map.append(probability_observed_rating)


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create 1 row, 2 columns subplot
    
    # First subplot for the normal distribution
    x_fixed = np.linspace(-10, 10, 1000)
    pdf_fixed = norm.pdf(x_fixed, dist_mean, std_dev)
    sns.lineplot(x=x_fixed, y=pdf_fixed, ax=axs[0])

    axs[0].fill_between(x_fixed, pdf_fixed, where=(x_fixed <= q_1), color='skyblue', alpha=0.5)
    axs[0].fill_between(x_fixed, pdf_fixed, where=(x_fixed > q_1) & (x_fixed <= q_2), color='lightgreen', alpha=0.5)
    axs[0].fill_between(x_fixed, pdf_fixed, where=(x_fixed > q_2) & (x_fixed <= q_3), color='salmon', alpha=0.5)
    axs[0].fill_between(x_fixed, pdf_fixed, where=(x_fixed > q_3) & (x_fixed <= q_4), color='wheat', alpha=0.5)
    axs[0].fill_between(x_fixed, pdf_fixed, where=(x_fixed > q_4), color='plum', alpha=0.5)
    axs[0].set_ylim(0, norm.pdf(0, 0, 1) * 1.1)  # Fixed y-axis for the normal distribution plot
    axs[0].set_title("Prior distribution of service quality")
    axs[0].set_ylabel("Probability Density")
    axs[0].set_xlabel("Experienced service quality")

    # Second subplot for the bar plot with matched colors
    bins = np.arange(1, 5.5, 0.5) 
    values = bin_map
    colors = ['skyblue', 'aquamarine', 'lightgreen', 'lightcoral', 'salmon', 'lightcoral', 'wheat', 'navajowhite', 'plum']  # Colors to match the normal distribution segments

    sns.barplot(x=bins, y=values, ax=axs[1], palette=colors)

    axs[1].set_ylim(0, 1)
    axs[1].set_title("Q\u0303 distribution")
    axs[1].set_ylabel("Probability density")
    axs[1].set_xlabel("Ratings [Rounded to nearest half integer]")

    # Adjust layout for better appearance
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)

    # Create a DataFrame
    df_barplot = pd.DataFrame({
        'Rounded Ratings ': bins,
        'Probability': values
    })

    df_normal_distribution = pd.DataFrame({
        'Ratings': [x+1 for x in range(len(distribution_prior.cut_points))],
        'Probability': [norm.cdf(upper_limit, loc=distribution_prior.dist_mean, scale=distribution_prior.std_dev) - 
                        norm.cdf(lower_limit, loc=distribution_prior.dist_mean, scale=distribution_prior.std_dev) 
                        for lower_limit, upper_limit in distribution_prior.cut_points]
                        })
    
    # Use st.columns to create a layout with 2 columns
    col_df_normal_distribution, col_df_barplot = st.columns(2)

    # Add titles and display tables in their respective columns
    with col_df_barplot:
        st.markdown("### P(Q\u0303 | N_jt, Q*)")
        st.table(df_barplot)

    with col_df_normal_distribution:
        st.markdown("### Prior distribution")
        st.table(df_normal_distribution)

else:
    st.sidebar.error("Please ensure that Q1 < Q2 < Q3 < Q4")

