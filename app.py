import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def main():
    with st.sidebar:
        st.title("Simulação TCL")
        block_op = st.selectbox('Selecione a distribuição: ', ['Binomial', 'Exponencial', 'Uniforme'])

        st.subheader("Opções:")
        if block_op == 'Binomial':
            p_binomial = st.number_input('Digite a probabilidade p (entre 0.01 e 0.99):', 0.01, 0.99, 0.5, step=0.01, format="%.2f")
            n_binomial = st.number_input('Digite o número total de tentativas (não negativo):', 1, 100, 100)

        elif block_op == 'Exponencial':
            lam = st.number_input('Digite o parâmetro lambda (positivo):', 1.0, 3.0, 1.0, step=0.1, format="%.2f")

        elif block_op == "Uniforme":
            a_uniforme = st.number_input('Digite o parâmetro a:', 0, 5, 0)
            b_uniforme = st.number_input('Digite o parâmetro b:', 5, 10, 5)

        numero_amostras = st.number_input('Digite a quantidade de amostras:', min_value=5, value=100)
        tamanho_amostra = st.number_input('Digite o tamanho das amostras:', min_value=1, value=30)


    if block_op == 'Binomial':
        # Criando o DataFrame para a distribuição binomial
        df_binomial = pd.DataFrame()
        for i in range(1, numero_amostras + 1):
            binomial_sample = np.random.binomial(n_binomial, p_binomial, size=tamanho_amostra)
            col = f'sample {i}'
            df_binomial[col] = binomial_sample

        # Ajustando o tamanho total da figura
        plt.figure(figsize=(12, 8))

        # Subplot 1: Histograma da distribuição binomial
        plt.subplot(1, 2, 1)
        sns.histplot(df_binomial.values.flatten(), kde=False, bins=30, color='purple', edgecolor='black')
        plt.title('Binomial Sample Distribution')
        plt.xlabel('Values')
        plt.ylabel('Sample Frequency')

        # Subplot 2: Distribuição das médias amostrais e ajuste da distribuição normal para binomial
        plt.subplot(1, 2, 2)
        df_binomial_sample_means = pd.DataFrame(df_binomial.mean(), columns=['Sample means'])
        # Ajuste o número de bins e a largura do bin conforme necessário
        sns.histplot(df_binomial_sample_means['Sample means'], kde=False, bins=20, color='purple', edgecolor='black', binwidth=0.2)

        mean_binomial_sample_means = df_binomial_sample_means['Sample means'].mean()
        se_binomial_sample_means = df_binomial_sample_means['Sample means'].std() / np.sqrt(tamanho_amostra)

        # Ajustando os eixos X
        x_binomial = np.linspace(df_binomial_sample_means['Sample means'].min(), df_binomial_sample_means['Sample means'].max(), 100)

        # Desenhando a distribuição normal ajustada para binomial
        plt.plot(x_binomial, norm.pdf(x_binomial, loc=mean_binomial_sample_means, scale=se_binomial_sample_means), 'r', label='Normal Distribution Fit (Binomial)')

        plt.title('Distribution of Sample Means (Binomial)')
        plt.xlabel('Sample Means')
        plt.ylabel('Frequency of Sample Sets')
        plt.legend()

        # Exibindo os plots no Streamlit
        st.pyplot(plt.gcf())

    elif block_op == 'Exponencial':
        df = pd.DataFrame()
        for i in range(1, numero_amostras + 1):
            exponential_sample = np.random.exponential(scale=1/lam, size=tamanho_amostra)
            col = f'sample {i}'
            df[col] = exponential_sample

        plt.figure(figsize=(12, 8))

        # Subplot 1: Histograma da distribuição exponencial
        plt.subplot(1, 2, 1)
        sns.histplot(df.values.flatten(), kde=False, bins=30, color='darkblue', edgecolor='black')
        plt.title('Sample Distribution')
        plt.xlabel('Values')
        plt.ylabel('Sample Frequency')

        # Subplot 2: Distribuição das médias amostrais e ajuste da distribuição normal
        plt.subplot(1, 2, 2)
        df_sample_means = pd.DataFrame(df.mean(), columns=['Sample means'])
        sns.histplot(df_sample_means['Sample means'], kde=False, bins=20, color='darkblue', edgecolor='black')

        mean_sample_means = df_sample_means['Sample means'].mean()
        se_sample_means = df_sample_means['Sample means'].std() / np.sqrt(tamanho_amostra)

        mu = 0
        sigma = 1

        # Ajustando os eixos X
        x = np.linspace(df_sample_means['Sample means'].min(), df_sample_means['Sample means'].max(), 100)

        # Desenhando a distribuição normal ajustada
        plt.plot(x, norm.pdf(x, loc=mean_sample_means, scale=se_sample_means), 'r', label='Normal Distribution Fit')

        plt.title('Distribution of Sample Means')
        plt.xlabel('Sample Means')
        plt.ylabel('Frequency of Sample Sets')
        plt.legend()

        # Exibindo os plots no Streamlit
        st.pyplot(plt.gcf())

    elif block_op == 'Uniforme':
        # Criando o DataFrame para a distribuição uniforme
        df_uniform = pd.DataFrame()
        for i in range(1, numero_amostras + 1):
            uniform_sample = np.random.uniform(low=a_uniforme, high=b_uniforme, size=tamanho_amostra)
            col = f'sample {i}'
            df_uniform[col] = uniform_sample

        # Ajustando o tamanho total da figura
        plt.figure(figsize=(12, 8))

        # Subplot 1: Histograma da distribuição uniforme
        plt.subplot(1, 2, 1)
        sns.histplot(df_uniform.values.flatten(), kde=False, bins=30, color='darkgreen', edgecolor='black')
        plt.title('Uniform Sample Distribution')
        plt.xlabel('Values')
        plt.ylabel('Sample Frequency')

        # Subplot 2: Distribuição das médias amostrais e ajuste da distribuição normal para uniforme
        plt.subplot(1, 2, 2)
        df_uniform_sample_means = pd.DataFrame(df_uniform.mean(), columns=['Sample means'])
        sns.histplot(df_uniform_sample_means['Sample means'], kde=False, bins=20, color='darkgreen', edgecolor='black')

        mean_uniform_sample_means = df_uniform_sample_means['Sample means'].mean()
        se_uniform_sample_means = df_uniform_sample_means['Sample means'].std() / np.sqrt(tamanho_amostra)

        # Ajustando os eixos X
        x_uniform = np.linspace(df_uniform_sample_means['Sample means'].min(), df_uniform_sample_means['Sample means'].max(), 100)

        # Desenhando a distribuição normal ajustada para uniforme
        plt.plot(x_uniform, norm.pdf(x_uniform, loc=mean_uniform_sample_means, scale=se_uniform_sample_means), 'r', label='Normal Distribution Fit (Uniform)')

        plt.title('Distribution of Sample Means (Uniform)')
        plt.xlabel('Sample Means')
        plt.ylabel('Frequency of Sample Sets')
        plt.legend()

        # Exibindo os plots no Streamlit
        st.pyplot(plt.gcf())

if __name__ == '__main__':
    main()
