import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import math

def main():
    with st.sidebar:
        st.title("Simulação TCL")
        block_op = st.selectbox('Selecione a distribuição: ', ['Binomial', 'Exponencial', 'Uniforme'])

        st.subheader("Opções:")
        if block_op == 'Binomial':
            p_binomial = st.number_input('Digite a probabilidade p (entre 0.01 e 0.99):', 0.01, 0.99, 0.5, step=0.01, format="%.2f")

        elif block_op == 'Exponencial':
            lambda_exponencial = st.number_input('Digite o parâmetro lambda (positivo):', 1.0, 3.0, 1.0, step=0.1, format="%.2f")

        elif block_op == "Uniforme":
            a_uniforme = st.number_input('Digite o parâmetro a:', 0, 5, 0)
            b_uniforme = st.number_input('Digite o parâmetro b:', 5, 10, 5)

        numero_amostras = st.number_input('Digite a quantidade de amostras:', 100)
        tamanho_amostra = st.number_input('Digite o tamanho das amostras:', 30)
    

    if block_op == 'Binomial':
        amostras_binomial = np.random.binomial(1, p_binomial, size=(numero_amostras, tamanho_amostra))
        medias_amostrais_binomial = np.mean(amostras_binomial, axis=1)

        # Ajustar o tamanho da figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # Ajuste o valor de figsize conforme necessário

        ax1.hist(amostras_binomial.flatten(), bins=30, density=True, alpha=0.5, color='b')
        ax1.set_title('Distribuição Binomial')

        ax2.hist(medias_amostrais_binomial, bins=30, density=True, alpha=0.5, color='b', label='Médias Amostrais')
        x = np.linspace(0, 1, 100)
        ax2.plot(x, norm.pdf(x, loc=p_binomial, scale=np.sqrt(p_binomial*(1-p_binomial)/tamanho_amostra)), 'r', label='Distribuição Normal Aproximada')
        ax2.set_title('Médias Amostrais - Binomial')
        ax2.legend()

        # Exibir os gráficos no Streamlit
        st.pyplot(fig)

    elif block_op == 'Exponencial':
        # Generate samples from the Exponential distribution
        amostras_exponencial = np.random.exponential(scale=1/lambda_exponencial, size=(numero_amostras, tamanho_amostra))

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.hist(amostras_exponencial.flatten(), bins=30, alpha=0.5, color='b', edgecolor='black', linewidth=1.2, label='Amostras')
        ax1.set_title('Distribuição Exponencial')
        ax1.set_xlabel('Valor')
        ax1.set_ylabel('Contagem')
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.hist(tamanho_amostra, bins=30, alpha=0.5, color='b', edgecolor='black', linewidth=1.2, label='Médias Amostrais')
        x = np.linspace(0, np.max(tamanho_amostra), 100)
        ax2.plot(x, norm.pdf(x, loc=1/lambda_exponencial, scale=np.sqrt(1/(lambda_exponencial**2 * tamanho_amostra))), 'r', label='Distribuição Normal Aproximada')
        ax2.set_title('Médias Amostrais - Exponencial')
        ax2.set_xlabel('Média Amostral')
        ax2.set_ylabel('Contagem')
        ax2.legend()
        st.pyplot(fig2)

    elif block_op == 'Uniforme':
        amostras_uniforme = np.random.uniform(a_uniforme, b_uniforme, size=(numero_amostras, tamanho_amostra))
        medias_amostrais_uniforme = np.mean(amostras_uniforme, axis=1)

        # Plotar histograma da distribuição uniforme
        fig1, ax1 = plt.subplots()
        ax1.hist(amostras_uniforme.flatten(), bins=30, density=True, alpha=0.5, color='b', label='Amostras')
        ax1.set_title('Distribuição Uniforme')
        ax1.legend()

        # Plotar histograma das médias amostrais e a distribuição normal aproximada
        fig2, ax2 = plt.subplots()
        ax2.hist(medias_amostrais_uniforme, bins=30, density=True, alpha=0.5, color='b', label='Médias Amostrais')
        x = np.linspace(np.min(medias_amostrais_uniforme), np.max(medias_amostrais_uniforme), 100)
        ax2.plot(x, norm.pdf(x, loc=(a_uniforme + b_uniforme) / 2, scale=np.sqrt(((b_uniforme - a_uniforme)**2) / (12 * tamanho_amostra))), 'r', label='Distribuição Normal Aproximada')
        ax2.set_title('Médias Amostrais - Uniforme')
        ax2.legend()

        # Exibir os gráficos no Streamlit
        st.pyplot(fig1)
        st.pyplot(fig2)

if __name__ == '__main__':
    main()
