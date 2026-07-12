
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from datetime import datetime, date
 
st.set_page_config(
    page_title="TFM Pricing Aéreo - Top 99%",
    page_icon="✈️",
    layout="wide"
)
 
APP_DIR = Path(__file__).parent
FECHA_CADUCIDAD = date.fromisoformat("2026-07-28")
 
hoy = datetime.today().date()
 
if hoy > FECHA_CADUCIDAD:
    st.error("Este enlace de visualización ha caducado.")
    st.info("La app estaba configurada para funcionar hasta el " + FECHA_CADUCIDAD.strftime("%d/%m/%Y") + ".")
    st.stop()
 
dias_restantes = (FECHA_CADUCIDAD - hoy).days
 
def existe(nombre):
    return (APP_DIR / nombre).exists()
 
@st.cache_data
def cargar_csv(nombre):
    ruta = APP_DIR / nombre
    if ruta.exists():
        return pd.read_csv(ruta)
    return pd.DataFrame()
 
def mostrar_mapa(nombre, altura=720):
    ruta = APP_DIR / nombre
 
    if not ruta.exists():
        st.warning("No se encuentra el archivo: " + nombre)
        return
 
    html = ruta.read_text(encoding="utf-8")
    components.html(html, height=altura, scrolling=True)
 
def mostrar_tabla(nombre, titulo):
    st.subheader(titulo)
 
    df = cargar_csv(nombre)
 
    if df.empty:
        st.warning("No se encuentra o está vacío el archivo: " + nombre)
        return
 
    columnas_numericas = df.select_dtypes(include="number").columns.tolist()
 
    col1, col2, col3 = st.columns([1, 1, 1])
 
    with col1:
        if "dificultad" in df.columns:
            opciones = ["Todas"] + sorted(df["dificultad"].dropna().unique().tolist())
            dificultad = st.selectbox("Dificultad", opciones, key="dif_" + nombre)
        else:
            dificultad = "Todas"
 
    with col2:
        if columnas_numericas:
            indice_mae = columnas_numericas.index("MAE") if "MAE" in columnas_numericas else 0
            ordenar_por = st.selectbox("Ordenar por", columnas_numericas, index=indice_mae, key="ord_" + nombre)
        else:
            ordenar_por = df.columns[0]
 
    with col3:
        ascendente = st.checkbox("Ascendente", value=False, key="asc_" + nombre)
 
    df_filtrado = df.copy()
 
    if dificultad != "Todas" and "dificultad" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["dificultad"] == dificultad]
 
    if ordenar_por in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values(ordenar_por, ascending=ascendente)
 
    st.dataframe(df_filtrado, use_container_width=True)
 
    st.download_button(
        "Descargar CSV",
        data=df_filtrado.to_csv(index=False).encode("utf-8"),
        file_name=nombre,
        mime="text/csv",
        key="download_" + nombre
    )
 
st.title("✈️ TFM Pricing Aéreo")
st.caption("Visualización de dificultad predictiva del modelo Top 99% de la Fase 4")
 
st.info(
    "Esta app estará activa hasta el **" +
    FECHA_CADUCIDAD.strftime("%d/%m/%Y") +
    "**. Quedan **" +
    str(dias_restantes) +
    " días**."
)
 
with st.sidebar:
    st.header("Información")
    st.write("Modelo: **Top 99% Fase 4**")
    st.write("Set evaluado: **Test**")
    st.write("Caducidad:", FECHA_CADUCIDAD.strftime("%d/%m/%Y"))
 
tabs = st.tabs([
    "🗺️ Mapas",
    "📍 Aeropuertos",
    "🏙️ Mercados",
    "🛫 Rutas aeropuerto",
    "🌐 Rutas mercado",
    "📊 Resúmenes",
    "📝 Texto memoria"
])
 
with tabs[0]:
    st.header("Mapas de dificultad predictiva")
 
    opcion = st.radio(
        "Selecciona visualización",
        [
            "Aeropuertos físicos de salida",
            "Mercados / ciudades de salida",
            "Rutas aeropuerto-aeropuerto",
            "Rutas mercado-mercado"
        ],
        horizontal=True
    )
 
    if opcion == "Aeropuertos físicos de salida":
        st.markdown("Distingue aeropuertos concretos: JFK, LGA, EWR, LAX, SFO, etc.")
        mostrar_mapa("mapa_aeropuertos_salida_dificultad_top99.html")
 
    elif opcion == "Mercados / ciudades de salida":
        st.markdown("Agrupa ciudades o mercados que pueden contener varios aeropuertos.")
        mostrar_mapa("mapa_mercados_salida_dificultad_top99.html")
 
    elif opcion == "Rutas aeropuerto-aeropuerto":
        st.markdown("Líneas verdes = rutas más fáciles; líneas rojas = rutas más difíciles.")
        mostrar_mapa("mapa_rutas_aeropuerto_dificultad_top99.html")
 
    elif opcion == "Rutas mercado-mercado":
        st.markdown("Agrupa las conexiones por mercado origen-destino mediante route_market_key.")
        mostrar_mapa("mapa_rutas_mercado_dificultad_top99.html")
 
with tabs[1]:
    st.header("Aeropuertos físicos de salida")
 
    mostrar_tabla(
        "lista_aeropuertos_salida_dificultad_top99.csv",
        "Lista completa de aeropuertos"
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        mostrar_tabla(
            "top_aeropuertos_dificiles_top99.csv",
            "Top aeropuertos más difíciles"
        )
 
    with col2:
        mostrar_tabla(
            "top_aeropuertos_faciles_top99.csv",
            "Top aeropuertos más fáciles"
        )
 
with tabs[2]:
    st.header("Mercados / ciudades de salida")
    st.markdown("Este análisis tiene en cuenta ciudades con varios aeropuertos.")
 
    mostrar_tabla(
        "lista_mercados_salida_dificultad_top99.csv",
        "Lista completa de mercados"
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        mostrar_tabla(
            "top_mercados_dificiles_top99.csv",
            "Top mercados más difíciles"
        )
 
    with col2:
        mostrar_tabla(
            "top_mercados_faciles_top99.csv",
            "Top mercados más fáciles"
        )
 
with tabs[3]:
    st.header("Rutas aeropuerto-aeropuerto")
 
    mostrar_tabla(
        "lista_rutas_aeropuerto_dificultad_top99.csv",
        "Lista completa de rutas aeropuerto-aeropuerto"
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        mostrar_tabla(
            "top_rutas_aeropuerto_dificiles_top99.csv",
            "Top rutas aeropuerto más difíciles"
        )
 
    with col2:
        mostrar_tabla(
            "top_rutas_aeropuerto_faciles_top99.csv",
            "Top rutas aeropuerto más fáciles"
        )
 
with tabs[4]:
    st.header("Rutas mercado-mercado")
 
    mostrar_tabla(
        "lista_rutas_mercado_dificultad_top99.csv",
        "Lista completa de rutas mercado-mercado"
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        mostrar_tabla(
            "top_rutas_mercado_dificiles_top99.csv",
            "Top rutas mercado más difíciles"
        )
 
    with col2:
        mostrar_tabla(
            "top_rutas_mercado_faciles_top99.csv",
            "Top rutas mercado más fáciles"
        )
 
with tabs[5]:
    st.header("Resúmenes agregados")
 
    mostrar_tabla(
        "resumen_aeropuertos_top99.csv",
        "Resumen por dificultad - Aeropuertos"
    )
 
    mostrar_tabla(
        "resumen_mercados_top99.csv",
        "Resumen por dificultad - Mercados"
    )
 
    mostrar_tabla(
        "resumen_rutas_aeropuerto_top99.csv",
        "Resumen por dificultad - Rutas aeropuerto"
    )
 
    mostrar_tabla(
        "resumen_rutas_mercado_top99.csv",
        "Resumen por dificultad - Rutas mercado"
    )
 
with tabs[6]:
    st.header("Texto interpretativo para memoria")
 
    ruta_texto = APP_DIR / "texto_memoria_mapas_dificultad_top99.txt"
 
    if ruta_texto.exists():
        texto = ruta_texto.read_text(encoding="utf-8")
        st.text_area("Texto generado", texto, height=500)
 
        st.download_button(
            "Descargar texto",
            data=texto.encode("utf-8"),
            file_name="texto_memoria_mapas_dificultad_top99.txt",
            mime="text/plain"
        )
    else:
        st.warning("No se encuentra el texto interpretativo.")
 
st.divider()
st.caption("TFM Pricing Aéreo · Modelo Top 99% Fase 4 · Visualización temporal de resultados")
