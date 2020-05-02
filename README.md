# rna-batalla-naval
RNA MultiPerceptrón Backpropagation que simula un jugador de batalla naval

# Configuración Local

## Imagen de docker
```bash
docker build --tag rna-batalla-naval .
```

## Corremos los comandos dentro del contenedor
```bash
docker run -it --rm --name batalla-naval -v "$PWD":/app rna-batalla-naval python ./src/nombre_archivo.py
```