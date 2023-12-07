import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def get_sample(dist, params):
    if dist == 'Binomial':
        return np.random.binomial(params['n'], params['p'], size=params['len'])
    elif dist == 'Exponential':
        return np.random.exponential(scale=1/(params['y']), size=params['len'])
    elif dist == 'Uniform':
        return np.random.uniform(low=params['a'], high=params['b'], size=params['len'])

def generate_samples(dist, params):
    df = pd.DataFrame()
    for i in range(1, params['qnt'] + 1):
        sample = get_sample(dist, params)
        col = f'sample {i}'
        df[col] = sample

    return df

def plot_graphics(distribution, df, params):
    plt.figure(figsize=(12, 8))

    # Subplot 1: Distribution Histogram
    plt.subplot(1, 2, 1)

    sns.histplot(df.values.flatten(), kde=False, bins=15, color='darkblue', edgecolor='black')

    plt.title(f'{distribution} Sample Distribution')
    plt.xlabel('Values')
    plt.ylabel('Sample Frequency')

    # Subplot 2: Sample means distribution
    plt.subplot(1, 2, 2)
    df_sample_means = pd.DataFrame(df.mean(), columns=['Sample means'])

    sns.histplot(df_sample_means['Sample means'], kde=False, bins=15, color='darkred', edgecolor='black')

    mean_sample_means = df_sample_means['Sample means'].mean()
    se_sample_means = df_sample_means['Sample means'].std() / np.sqrt(params['len'])

    x_sample = np.linspace(df_sample_means['Sample means'].min(), df_sample_means['Sample means'].max(), 100)

    plt.title(f'Distribution of Sample Means ({distribution})')
    plt.xlabel('Sample Means')
    plt.ylabel('Frequency of Sample Sets')
    plt.legend()

    st.pyplot(plt.gcf())


def main():
    with st.sidebar:
        st.title("CLT Simulation")
        block_op = st.selectbox('Select a distribution: ', ['Binomial', 'Exponential', 'Uniform'])
        params = {}

        st.subheader("Options:")
        if block_op == 'Binomial':
            params['p'] = st.number_input('Enter the probability p (0.01 <= p <= 0.99):', 0.01, 0.99, 0.5, step=0.01, format="%.2f")
            params['n'] = st.number_input('Enter the total number of trials (n >= 1):', min_value=1, value=20)

        elif block_op == 'Exponential':
            params['y'] = st.number_input('Enter the lambda parameter (λ > 0):', min_value=0.1, value=1.0, step=0.1, format="%.2f")

        elif block_op == "Uniform":
            params['a'] = st.number_input('Enter the parameter a (0 <= a <= b):', min_value=0, value=0)
            params['b'] = st.number_input('Enter the parameter b (b >= a):', min_value=0, value=5)

            if params['b'] < params['a']:
                st.error('The value of b must be greater than or equal to a.')

        params['qnt'] = st.number_input('Enter the number of samples:', min_value=5, value=100)
        params['len'] = st.number_input('Enter the size of the samples:', min_value=1, value=30)

    df = generate_samples(block_op, params)
    plot_graphics(block_op, df, params)

    plt.subplots_adjust(wspace=0.3)

if __name__ == '__main__':
    main()
