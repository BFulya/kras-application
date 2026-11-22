# KRAS kuantum-klasik uretici model - calistirma ortami
#
# Neden Docker?
#   Windows'ta Smart App Control, pip ile gelen imzasiz native eklentileri
#   (qiskit._accelerate, rdkit.rdBase) blokluyor. Linux konteyneri bu
#   politikadan etkilenmez; Windows guvenlik ayarlarina dokunmadan
#   qiskit + rdkit'i tam calisir halde kullanmamizi saglar.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# rdkit ve matplotlib'in ihtiyac duydugu sistem kutuphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bagimliliklari once kur: kod degisince katman yeniden kullanilir
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Test bagimliliklari ayri katmanda: calistirma bagimliliklarinin agir
# onbellek katmani (torch, qiskit) bozulmadan guncellenebilir
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .

CMD ["python", "-c", "import qiskit, rdkit, selfies, torch; print('Ortam hazir:', qiskit.__version__)"]
