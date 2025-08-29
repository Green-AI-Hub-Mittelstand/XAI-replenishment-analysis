# 1. Use the official Python 3.12 slim image as a base
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
# This is done first to leverage Docker's layer caching for faster builds
COPY requirements.txt .

# 4. Install the Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your application files and folders into the container
# This includes app.py, assets/, data/, pages/, and saved_models/
COPY . .

# 6. Expose the port that Dash runs on (default is 8050)
EXPOSE 8050

# 7. Define the command to run the application using gunicorn
# The --bind flag makes the app accessible from outside the container
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]