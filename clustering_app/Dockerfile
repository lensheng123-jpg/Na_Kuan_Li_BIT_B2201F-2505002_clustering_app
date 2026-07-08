FROM python:3.12-slim

WORKDIR /app

# Install specific versions known to work together
RUN pip install --no-cache-dir \
    umap-learn==0.5.5 \
    numba==0.59.1 \
    llvmlite==0.42.0

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]