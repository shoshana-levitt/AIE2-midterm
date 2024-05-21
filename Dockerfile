# Use the official Python image from the Docker Hub
FROM python:3.9

# Create a user with the UID 1000
RUN useradd -m -u 1000 user

# Switch to the user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the app directory
WORKDIR $HOME/app

# Copy requirements file first to leverage Docker cache
COPY --chown=user requirements.txt .

# Update pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in batches to identify slow packages
RUN pip install --no-cache-dir chainlit==0.7.700 cohere==4.37 openai tiktoken
RUN pip install --no-cache-dir python-dotenv==1.0.0 langchain langchain-openai
RUN pip install --no-cache-dir langchain_core langchain-community langchainhub ragas
RUN pip install --no-cache-dir qdrant-client pymupdf pandas

# Copy the rest of the application code
COPY --chown=user . .

# Set the entrypoint
ENTRYPOINT ["chainlit", "run", "app.py", "--port", "7860"]
