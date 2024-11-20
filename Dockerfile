# Use the NVIDIA CUDA base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy the current directory contents into the container at /app
COPY requirements.txt /app
# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
model_name = 'intfloat/multilingual-e5-large'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModel.from_pretrained(model_name)"

COPY . /app

# Run both scripts concurrently
CMD ["bash", "-c", "python3 basic_html_server_embeder.py & python3 db.py"]