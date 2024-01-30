# Use the official Jupyter Docker image as the base image
FROM jupyter/base-notebook

# Set the working directory to /app
WORKDIR /home/jovyan/work/SensoriumDecoding

# Copy the poetry files to the container
COPY pyproject.toml poetry.lock ./

# Install poetry and project dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
