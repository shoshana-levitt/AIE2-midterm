FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]

# # Use the official Python image from the Docker Hub
# FROM python:3.9
#
# # Install system dependencies if needed (uncomment if required)
# # RUN apt-get update && apt-get install -y some-package
#
# # Create a user with the UID 1000
# RUN useradd -m -u 1000 user
#
# # Switch to the user
# USER user
#
# # Set environment variables
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH
#
# # Create the app directory
# WORKDIR $HOME/app
#
# # Copy requirements file first to leverage Docker cache
# COPY --chown=user requirements.txt .
#
# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt
#
# # Copy the rest of the application code
# COPY --chown=user . .
#
# # Set the entrypoint
# ENTRYPOINT ["chainlit", "run", "app.py", "--port", "7860"]
