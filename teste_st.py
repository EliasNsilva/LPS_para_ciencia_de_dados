import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with st.form(key='my_form'):
    with st.sidebar:
        st.title("Linha de produto de software para Ciencia de dados")
        file = st.file_uploader("Upload do arquivo CSV", type="csv")

        # Verifica se o usuário fez o upload do arquivo
        if file is not None:
            df = pd.read_csv(file)

        line_plot = st.checkbox('Grafico de linha')
        bar_plot = st.checkbox('Grafico de barras')
        scatter_plot = st.checkbox('Grafico de dispersão')
        pair_plot = st.checkbox('Grafico de pares')
        st.form_submit_button("Submit")

if file is not None:
    st.subheader('Tabela de dados')
    st.dataframe(df)

if line_plot:
    st.subheader('Grafico de linha')
    x = st.selectbox("Escolha a coluna para o eixo x", df.columns)
    y = st.multiselect("Escolha a coluna para o eixo y", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], label=y)
    plt.legend()
    st.pyplot(fig)

if bar_plot:
    st.subheader('Grafico de barras')
    x = st.selectbox("Escolha a coluna para o eixo x", df.columns)
    y = st.multiselect("Escolha a coluna para o eixo y", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.bar(df[x], df[y], label=y)
    plt.legend()
    st.pyplot(fig)

if scatter_plot:
    st.subheader('Grafico de dispersão')
    x = st.selectbox("Escolha a coluna para o eixo x", df.columns)
    y = st.selectbox("Escolha a coluna para o eixo y", df.columns)
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], label=y)
    st.pyplot(fig)
    #st.scatter_chart(df, x=x, y=y)

if pair_plot:
    st.subheader('Grafico de pares')
    hue = st.selectbox("Escolha a coluna para o hue", df.columns)
    fig = plt.figure(figsize=(10, 6))
    sns.pairplot(df, hue=hue)
    st.pyplot(fig)

#st.sidebar.markdown("# Sidebar")

#select_event = st.sidebar.selectbox('Escolha o evento',
#                                    ['Teste 1', 'Teste 2'])

#st.write(select_event)

# st.sidebar.markdown("""
# Example times in the H1 detector:
# * 1126259462.4    (GW150914) 
# * 1187008882.4    (GW170817) 
# * 1128667463.0    (hardware injection)
# * 1132401286.33   (Koi Fish Glitch) 
# """)

#dtboth = st.sidebar.slider('Time Range (seconds)', 0.1, 8.0, 1.0)

# x = st.selectbox("Escolha a coluna para o eixo x", df.columns)

# y = st.multiselect("Escolha a coluna para o eixo y", df.columns)

#st.line_chart(df, x=x, y=y)

# fig = plt.figure(figsize=(10, 6))
# sns.pairplot(df)
# st.pyplot(fig)
# plt.show()
#insira o link para a base de dados com streamlit


