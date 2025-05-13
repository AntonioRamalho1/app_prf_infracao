import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gdown
import tempfile
import os
import joblib
import psutil

@st.cache_data
def get_km_range_from_pipeline(_pipeline):   # ← underline no nome
    try:
        coltrans = _pipeline.named_steps['prep']
        scaler   = coltrans.named_transformers_['scale_num']
        num_cols = coltrans.transformers_[0][2]
        idx      = num_cols.index('km_infracao')
        min_km   = int(scaler.data_min_[idx])
        max_km   = int(scaler.data_max_[idx])
        return min_km, max_km
    except Exception as e:
        st.warning(f"Não achei range no scaler: {e}")
        return 0, 10000

def add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['hora_sin'] = np.sin(2*np.pi*out['hora_infracao']/24)
    out['hora_cos'] = np.cos(2*np.pi*out['hora_infracao']/24)
    out['mes_sin']  = np.sin(2*np.pi*out['mes']/12)
    out['mes_cos']  = np.cos(2*np.pi*out['mes']/12)
    out['dia_semana_sin'] = np.sin(2*np.pi*out['dia_semana']/7)
    out['dia_semana_cos'] = np.cos(2*np.pi*out['dia_semana']/7)
    return out.drop(columns=['hora_infracao','mes','dia_semana'])
    
# ==============================================
# 🎨 CONFIGURAÇÕES INICIAIS E DESIGN DA PÁGINA
# ==============================================
st.set_page_config(
    page_title="Previsor de Infrações PRF",
    page_icon="🚔",
    layout="centered",
    initial_sidebar_state="expanded"
)



# Custom CSS para melhorar a aparência
# Custom CSS
st.markdown("""
<style>
/* ================== VARIÁVEIS DE COR ================== */
:root {
    --primary:   #FF5733; /* laranja vivo */
    --secondary: #C70039; /* vermelho escuro */
    --text-dark: #212121; /* quase preto */
    --text-light: #FFFFFF; /* branco */
}

/* ================== LAYOUT GERAL ================== */
.stApp {
    background-color: #f5f5f5;
    color: var(--text-dark);
    font-family: "Helvetica", sans-serif;
}

/* ================== TÍTULOS ================== */
h1 {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 10px;
}
h2 {
    color: var(--secondary) !important;
    margin-top: 1.5rem !important;
}

/* ================== SIDEBAR ================== */
[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

/* ================== WIDGETS ================== */
label, .stSelectbox, .stSlider, .stNumberInput, .stTextInput {
    color: var(--text-dark) !important;
    font-weight: 500;
}

/* ================== BOTÕES GERAIS ================== */
.stButton>button {
    background-color: var(--primary) !important;
    color: var(--text-light) !important;
    border: none;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    margin: 0.5rem 0;
    font-weight: 600;
    transition: all 0.25s ease;
    box-shadow: 0 3px 5px rgba(0,0,0,0.15);
}
.stButton>button:hover {
    background-color: var(--secondary) !important;
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 6px 10px rgba(0,0,0,0.25);
}
.stButton>button:focus,
.stButton>button:active {
    outline: none;
    box-shadow: 0 0 0 3px rgba(199,0,57,0.4);
}
.stButton > button:disabled {
    background-color: var(--primary) !important;
    color: var(--text-light) !important;
    opacity: 0.6 !important;
    cursor: not-allowed;
}

/* ================== BOTÃO SUBMIT DO FORMULÁRIO ================== */
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background-color: var(--primary) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    padding: 12px 28px !important;
    border-radius: 8px !important;
    border: 2px solid var(--primary) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    background-color: var(--secondary) !important;
    color: white !important;
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}
button[data-testid="stBaseButton-secondaryFormSubmit"] * {
    color: white !important;
    opacity: 1 !important;
    font-weight: bold !important;
    visibility: visible !important;
}

/* ================== RADIO BUTTONS ================== */
.stRadio [class*="st-"] label {
    color: black !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-weight: 500 !important;
}
div[role="radio"][aria-checked="true"] label {
    color: black !important;
    opacity: 1 !important;
    font-weight: bold !important;
}
div[role="radio"][aria-checked="false"] label {
    color: black !important;
    opacity: 1 !important;
}
.stRadio [role="radiogroup"] {
    gap: 1rem;
}
.stRadio [role="radio"] {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.stRadio label,
.stRadio div[role="radio"] > label > div {
    color: black !important;
    font-weight: 500;
}
.stRadio div[role="radio"] > label > div:first-child {
    background-color: white !important;
    border: 2px solid var(--primary) !important;
}
.stRadio div[role="radio"][aria-checked="true"] > label > div:first-child > div {
    background-color: var(--primary) !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: black !important;
    opacity: 1 !important;
    font-weight: 500 !important;
    visibility: visible !important;
}

/* ================== MENSAGENS E RESULTADOS ================== */
.stAlert, .stMarkdown>div {
    color: var(--text-dark) !important;
    font-weight: 600;
    border-radius: 8px !important;
}
.result-container {
    background-color: var(--text-light);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.stAlert[data-baseweb="notification"]:has(.stMarkdown) {
    background-color: #C8E6C9 !important;
    border: 1px solid #2E7D32 !important;
    color: #1B5E20 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ==============================================
# Função de exibição do cabeçalho
# ==============================================
def show_header():
    st.title("Previsor de Infrações PRF")
    st.markdown("### Aplicativo para previsão de infrações rodoviárias com base em dados históricos da PRF.")
# ==============================================
# 📥 CARREGAMENTO DE MODELO E DADOS
# ==============================================
# 📥 CARREGAMENTO DE MODELO E DADOS
@st.cache_resource(max_entries=1)
def load_pipeline():
    try:
        # Novo ID do arquivo do Google Drive
        file_id = "1ek_Em7R8HEbjL9jRtTg8uB8fZmumVudg"
        url = f"https://drive.google.com/uc?id={file_id}"

        # Cria um caminho temporário adequado para Windows/Linux/Mac
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "pipeline.pkl")

        gdown.download(url, output_path, quiet=False)

        with open(output_path, "rb") as f:
            obj = joblib.load(f)

        return obj['pipeline'], obj['base_cols']

    except Exception as e:
        st.error(f"Erro ao carregar o pipeline: {e}")
        return None, None

# ==============================================
# 👥 INTERFACE DE ENTRADA DE DADOS
# ==============================================
# Modificando a interface de entrada de dados
# Modificando a interface de entrada de dados
def input_data_interface(km_range):
    st.markdown("### 📝 Inserir Dados para Previsão")

    dados = {}
    min_km, max_km = km_range
    with st.form("input_form"):

        # ---------- CAMPOS PRINCIPAIS ----------

        dados['km_infracao'] = st.slider(
        "KM da infração",
        min_value=min_km,
        max_value=max_km,
        value=(min_km + max_km) // 2,
        step=1
        )
    
        dados['hora_infracao'] = st.selectbox(
            "Hora da infração",
            np.arange(0, 24),
            format_func=lambda x: f"{x:02d}:00"
        )

        dados['mes'] = st.selectbox(
            "Mês da infração", np.arange(1, 13)
        )

        dias_semana = {
            'Segunda':0, 'Terça':1, 'Quarta':2,
            'Quinta':3, 'Sexta':4, 'Sábado':5, 'Domingo':6
        }
        dados['dia_semana'] = dias_semana[
            st.selectbox("Dia da semana", list(dias_semana.keys()))
        ]

        dados['br_infracao'] = st.number_input(
            "Número da BR", min_value=0, max_value=9999,
            value=230, step=1, format="%d"
        )

        dados['feriado'] = 1 if st.selectbox(
            "Foi feriado?", ["Não", "Sim"], index=0
        ) == "Sim" else 0

        with st.expander("⚙️ Opções avançadas (geralmente deixe padrão)", expanded=False):

            abordagem = st.radio("O veículo foi abordado pela PRF?", ["Não","Sim"], horizontal=True)
            dados['indicador_de_abordagem'] = 1 if abordagem == "Sim" else 0

            assinatura = st.radio("O condutor assinou o Auto de Infração?", ["Não","Sim"], horizontal=True)
            dados['assinatura_do_auto'] = 1 if assinatura == "Sim" else 0

            estrangeiro = st.radio("O veículo possui placa estrangeira?", ["Não","Sim"], horizontal=True)
            dados['indicador_veiculo_estrangeiro'] = 1 if estrangeiro == "Sim" else 0

            sentido = st.radio("Sentido do tráfego na via:", ["Decrescente (km ↓)","Crescente (km ↑)"], horizontal=True)
            dados['sentido_trafego'] = 0 if "Decrescente" in sentido else 1

        # ---------- BOTÃO SUBMIT ----------
        submitted = st.form_submit_button("🔮 Realizar Previsão")

    return dados, submitted

# ==============================================
# 📊 VISUALIZAÇÃO DE RESULTADOS
# ==============================================
# Modificando a parte onde a previsão é feita
def show_results(pipeline, dados, base_cols):
    try:
        df = pd.DataFrame([dados])
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[base_cols]

        pred = pipeline.predict(df)[0]
        st.success(f"### Infração prevista: **{pred}**")

        proba = pipeline.predict_proba(df)[0]
        top3 = (pd.Series(proba, index=pipeline.classes_)
                   .nlargest(3)
                   .reset_index()
                   .rename(columns={'index':'Infração',0:'Probabilidade'}))
        st.dataframe(top3, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar previsão: {e}")

# ==============================================
# 📤 UPLOAD DE ARQUIVO CSV
# ==============================================
def file_upload_section(modelo, X_full):
    st.markdown("### 📤 Carregar Arquivo CSV")
    
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV com os dados para previsão",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Verificar colunas
            if set(df.columns) != set(X_full.columns):
                missing = set(X_full.columns) - set(df.columns)
                extra = set(df.columns) - set(X_full.columns)
                error_msg = "Erro: As colunas do arquivo não correspondem ao modelo."
                if missing:
                    error_msg += f"\n\nColunas faltantes: {', '.join(missing)}"
                if extra:
                    error_msg += f"\n\nColunas extras: {', '.join(extra)}"
                st.error(error_msg)
                return
            
            # Processar previsões
            with st.spinner("Processando arquivo..."):
                previsoes = modelo.predict(df)
                df_resultado = df.copy()
                df_resultado["Previsão"] = previsoes
                
                st.success(f"Previsões concluídas para {len(df)} registros!")
                st.dataframe(df_resultado, use_container_width=True)
                
                # Opção para download
                csv = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download dos Resultados",
                    data=csv,
                    file_name="resultados_previsao.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

# ==============================================
# ℹ️ SEÇÃO DE INFORMAÇÕES SOBRE O MODELO
# ==============================================
def model_info_section():
    st.sidebar.markdown("## ℹ️ Sobre o Modelo")
    
    with st.sidebar.expander("📚 Como funciona", expanded=True):
        st.write("""
        Este modelo preditivo utiliza algoritmo **Random Forest** para classificar o tipo de infração mais provável 
        em rodovias federais com base em dados históricos da PRF.
        
        **Funcionalidades:**
        - Previsão individual via formulário
        - Processamento em lote via arquivo CSV
        - Visualização detalhada das probabilidades
        """)
    
    with st.sidebar.expander("⚙️ Métricas do Modelo", expanded=False):
        st.write("""
        **Desempenho Médio (5.1 Milhões de dados):**
        - Acurácia geral: 78,94%
        - Precisão média ponderada: 74,17%
        - Recall médio ponderado: 78,94%
        - F1-score médio ponderado: 72,79%
        
        **Variáveis mais importantes:**
        1. Velocidade do veículo
        2. Tipo de rodovia
        3. Horário do dia
        4. Condições climáticas
        """)
    
    with st.sidebar.expander("⚠️ Limitações", expanded=False):
        st.write("""
        - Baseado em dados históricos (2024)
        - Não considera eventos atípicos
        - Precisão pode variar por região
        - Requer atualização periódica
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Desenvolvido por:**\n\nAntonio Ramalho e Allan Victor - CDN")

# ==============================================
# 📌 FUNÇÃO PRINCIPAL
# ==============================================
def print_memory_usage():
    process = psutil.Process(os.getpid())
    st.write(f"🧠 Memória usada: {process.memory_info().rss / 1024 ** 2:.2f} MB")

print_memory_usage()

def main():
    show_header()

    model_info_section()

    tab1, tab2 = st.tabs(["🔮 Previsão Individual", "📁 Processar Arquivo"])

    with tab1:
        km_range = (0, 1000)  # Valores padrão
        dados, submitted = input_data_interface(km_range)
        if submitted:
            # Carrega o modelo apenas se necessário
            pipeline, base_cols = load_pipeline()
            if pipeline is None:
                return
            show_results(pipeline, dados, base_cols)

    with tab2:
        uploaded = st.file_uploader("CSV para previsão", type="csv")
        if uploaded:
            pipeline, base_cols = load_pipeline()
            if pipeline is None:
                return
            df = pd.read_csv(uploaded)
            if set(df.columns) != set(base_cols):
                st.error("Colunas do CSV não batem com o modelo.")
            else:
                prev = pipeline.predict(df)
                df['Previsão'] = prev
                st.dataframe(df, use_container_width=True)
                st.download_button(...)

    st.markdown("---")
    st.caption("Dados PRF 2024 • Antonio Ramalho & Allan Victor")

if __name__ == "__main__":
    main()
