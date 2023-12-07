%%writefile app.py

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
    samples = []

    for i in range(1, params['qnt'] + 1):
        sample = get_sample(dist, params)
        samples.append(pd.Series(sample, name=f'sample {i}'))

    df = pd.concat(samples, axis=1)

    return df

def plot_normal(ax, sample_means, actual_mean, actual_standard_deviation):
    x = np.linspace(min(sample_means), max(sample_means), 100)
    rv = norm(loc=actual_mean, scale=actual_standard_deviation)
    y_normal = rv.pdf(x)

    ax2 = ax.twinx()
    ax2.plot(x, y_normal, 'green', linewidth=2, label='Normal Distribution')
    ax2.set_ylabel('Normal Distribution', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

def plot_graphics(distribution, df, params):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Subplot 1: Distribution Histogram
    axes[0].hist(df.values.flatten(), bins=15, color='darkblue', edgecolor='black')
    axes[0].set_title(f'{distribution} Sample Distribution')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Sample Frequency')

    # Subplot 2: Sample means distribution
    df_sample_means = pd.DataFrame(df.mean(), columns=['Sample means'])
    axes[1].hist(df_sample_means['Sample means'], bins=15, color='darkred', edgecolor='black')

    sample_means = df_sample_means['Sample means'].mean()
    se_sample_means = df_sample_means['Sample means'].std() / np.sqrt(params['len'])

    x_sample = np.linspace(df_sample_means['Sample means'].min(), df_sample_means['Sample means'].max(), 100)

    axes[1].set_title(f'Distribution of Sample Means ({distribution})')
    axes[1].set_xlabel('Sample Means')
    axes[1].set_ylabel('Frequency of Sample Sets')

    # Plotar distribuição normal sobreposta com legenda
    plot_normal(axes[1], df_sample_means['Sample means'], sample_means, se_sample_means)

    # Ajustar layout
    plt.subplots_adjust(wspace=0.3)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)


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

        params['qnt'] = st.number_input('Enter the number of samples (n >= 1):', min_value=1, value=100)
        params['len'] = st.number_input('Enter the size of the samples (n >= 1):', min_value=1, value=30)

    if block_op == 'Binomial':
        st.markdown("""
        <h3>Binomial:</h3>
        <ul>
            <li>The distribution of sample means approaches a normal distribution, as expected by the Central Limit Theorem (CLT).</li>
            <li>As n increases, the convergence to the normal distribution becomes more evident.</li>
        </ul><br>
        """, unsafe_allow_html=True)
    elif block_op == 'Exponential':
        st.markdown("""
        <h3>Exponential:</h3>
        <ul>
            <li>Similar to the binomial distribution, the distribution of sample means converges to a normal distribution as n increases.</li>
            <li>The shape of the exponential distribution affects the rate of convergence.</li>
        </ul><br>
        """, unsafe_allow_html=True)
    elif block_op == 'Uniform':
        st.markdown("""
        <h3>Uniform:</h3>
        <ul>
            <li>The uniform distribution, being less affected by long tails, may converge to a normal distribution more quickly.</li>
            <li>The convergence of sample means to a normal distribution is faster as the sample size increases.</li>
        </ul><br>
        """, unsafe_allow_html=True)

    df = generate_samples(block_op, params)
    plot_graphics(block_op, df, params)

    st.markdown("""<h4>Alunos:</h4>
        <h5>Caroline Campos Carvalho e Iago Zagnoli Albergaria</h4>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
