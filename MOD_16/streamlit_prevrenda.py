
import streamlit as st
st.set_page_config(
    page_title="Previsão de Renda",
    page_icon="💰",
    layout="wide",
)

#  Importações
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#  Estilo dos gráficos
sns.set(context='talk', style='ticks')

#  Título e descrição
st.title(" Previsão de Renda")
st.markdown("""
Este aplicativo utiliza um modelo de machine learning para prever a **renda mensal estimada** com base em características individuais.
Preencha os campos no menu lateral e clique em **Prever** para obter o resultado.
""")

#  Carregar dados
renda = pd.read_csv('previsao_de_renda.csv')

#  Gráficos ao longo do tempo
st.write('## Gráficos ao longo do tempo')
fig1, ax = plt.subplots(8, 1, figsize=(10, 70))
renda[['posse_de_imovel', 'renda']].plot(kind='hist', ax=ax[0])
sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda, ax=ax[1])
sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda, ax=ax[2])
sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, ax=ax[3])
sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda, ax=ax[4])
sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda, ax=ax[5])
sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax[6])
sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda, ax=ax[7])
for a in ax[1:]:
    a.tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(fig1)

#  Gráficos bivariados
st.write('## Gráficos bivariada')
fig2, ax2 = plt.subplots(7, 1, figsize=(10, 50))
sns.barplot(x='posse_de_imovel', y='renda', data=renda, ax=ax2[0])
sns.barplot(x='posse_de_veiculo', y='renda', data=renda, ax=ax2[1])
sns.barplot(x='qtd_filhos', y='renda', data=renda, ax=ax2[2])
sns.barplot(x='tipo_renda', y='renda', data=renda, ax=ax2[3])
sns.barplot(x='educacao', y='renda', data=renda, ax=ax2[4])
sns.barplot(x='estado_civil', y='renda', data=renda, ax=ax2[5])
sns.barplot(x='tipo_residencia', y='renda', data=renda, ax=ax2[6])
sns.despine()
st.pyplot(fig2)

#  Previsão de Renda
st.write('##  Previsão de Renda com Modelo Treinado')

# Carregar modelo
try:
    modelo = joblib.load("modelo_previsao.pkl")
    variaveis = joblib.load("variaveis_modelo.pkl")  # lista de colunas usadas no modelo
except FileNotFoundError:
    st.error(" Arquivo do modelo não encontrado. Verifique se 'modelo_previsao.pkl' e 'variaveis_modelo.pkl' estão na pasta.")
    st.stop()

# Interface de entrada
with st.form("formulario_renda"):
    st.write("### Preencha os dados abaixo para prever a renda:")

    posse_de_imovel = st.selectbox("Possui imóvel?", ['Sim', 'Não'])
    posse_de_veiculo = st.selectbox("Possui veículo?", ['Sim', 'Não'])
    qtd_filhos = st.slider("Quantidade de filhos", 0, 10, 0)
    tipo_renda = st.selectbox("Tipo de renda", renda['tipo_renda'].unique())
    educacao = st.selectbox("Escolaridade", renda['educacao'].unique())
    estado_civil = st.selectbox("Estado civil", renda['estado_civil'].unique())
    tipo_residencia = st.selectbox("Tipo de residência", renda['tipo_residencia'].unique())
    tempo_emprego = st.number_input("Tempo de emprego (anos)", min_value=0.0, step=0.1)
    qt_pessoas_residencia = st.slider("Quantidade de pessoas na residência", 1, 10, 1)

    enviar = st.form_submit_button("Prever Renda")

#  Previsão
if enviar:
    entrada = pd.DataFrame([{
        'posse_de_imovel': 1 if posse_de_imovel == 'Sim' else 0,
        'posse_de_veiculo': 1 if posse_de_veiculo == 'Sim' else 0,
        'qtd_filhos': qtd_filhos,
        'tipo_renda': tipo_renda,
        'educacao': educacao,
        'estado_civil': estado_civil,
        'tipo_residencia': tipo_residencia,
        'tempo_emprego': tempo_emprego,
        'qt_pessoas_residencia': qt_pessoas_residencia
    }])

    entrada = pd.get_dummies(entrada)
    for col in variaveis:
        if col not in entrada.columns:
            entrada[col] = 0
    entrada = entrada[variaveis]

    try:
        pred = modelo.predict(entrada)[0]
        st.success(f"💰 Renda estimada: R$ {pred:,.2f}")
    except Exception as e:
        st.error(f"Erro ao fazer previsão: {e}")


