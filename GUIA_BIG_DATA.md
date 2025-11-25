# GUÍA DE USO: BIG DATA EN LYLY_PREDICCION

Esta guía explica cómo ejecutar y aprovechar la funcionalidad de Big Data (GDELT) integrada en el proyecto.

## 📋 Prerrequisitos

Asegúrate de tener el entorno activado y las dependencias instaladas:

```bash
# Activar entorno virtual (si no está activo)
# Windows:
.venv\Scripts\activate

# Instalar dependencias (incluye duckdb, requests, tqdm)
pip install -r requirements.txt
pip install tqdm
```

---

## 🚀 1. Verificación Rápida (Test)

Antes de lanzarte a descargar gigabytes de datos, verifica que todo funcione con una prueba pequeña.

Ejecuta el script de verificación:
```bash
python verify_big_data.py
```
**Resultado esperado:**
- Descarga de 1 o 2 días de datos.
- Descompresión automática.
- Procesamiento con DuckDB.
- Un mensaje final: `✅ PRUEBA DE BIG DATA EXITOSA`.

---

## 🛠️ 2. Uso Normal (Integrado en el Menú)

Para el uso diario o demostraciones rápidas:

1. Ejecuta el programa principal:
   ```bash
   python prediccion_meta_modular.py
   ```
2. Selecciona la **Opción 7 (Big Data Pipeline)**.
   - Esto descargará los últimos 5-7 días de noticias globales.
   - Filtrará noticias sobre META/Facebook/Zuckerberg.
   - Generará el gráfico de impacto (`impacto_noticias_meta.png`).

3. Selecciona la **Opción 3 (Entrenar Modelo)**.
   - El modelo detectará automáticamente los datos de sentimiento.
   - Entrenará el Random Forest usando el sentimiento de ayer (`Sentiment_Lag1`) para predecir el precio de hoy.
   - Te mostrará la **Feature Importance** al final.

---

## 🌍 3. FASE 4: La Gran Descarga (Historial Completo)

> **Objetivo:** Descargar 1 año completo de datos (aprox. 100GB descomprimidos, procesados en streaming) para obtener una "Feature Importance" real distinta de 0.0.

Hemos creado un script dedicado para esta tarea pesada, ya que puede tardar varias horas dependiendo de tu internet.

### Pasos para la Gran Descarga:

1. **Asegúrate de tener espacio en disco:** Necesitarás al menos 10-20GB libres temporalmente (los ZIPs se descargan y descomprimen).
2. **Ejecuta el script de historial:**
   ```bash
   python run_full_history.py
   ```
   *Este script mostrará una barra de progreso y manejará errores de conexión automáticamente.*

3. **Espera a que termine:**
   - El script descargará 365 días de archivos GDELT.
   - Procesará todo con DuckDB.
   - Guardará el resultado final en un archivo ligero: `gdelt_history_1year.csv`.

4. **Usar el historial en el modelo:**
   - Una vez tengas `gdelt_history_1year.csv`, el sistema principal podrá cargarlo (requiere una pequeña modificación en `load_and_process_data` para leer este CSV si existe, en lugar de descargar solo 5 días).

---

## ❓ Preguntas Frecuentes

**¿Por qué la importancia del sentimiento me sale 0.0000?**
Porque en la prueba estándar solo usamos 5 días de noticias contra años de precios históricos. Para el modelo, el sentimiento es "neutral" (0) casi siempre. Al ejecutar la **Fase 4 (Gran Descarga)**, llenarás esos huecos y el modelo empezará a aprender patrones reales.

**¿Dónde están los archivos descargados?**
En la carpeta `gdelt_data/`. Puedes borrarlos si necesitas espacio, pero tendrás que descargarlos de nuevo si quieres reprocesar.

**¿Qué es DuckDB?**
Es el motor que nos permite procesar esos cientos de archivos CSV/ZIP en segundos sin que tu memoria RAM explote (Out-of-Core Processing).
