#Use a slim Python image for a smaller container size

FROM python:3.13.9-slim

#Set working directory

WORKDIR /app

#Install the pinned dependencies (requirements.txt is the single source of truth)

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#Copy model artifacts and application code
#NOTE: You must run models.py locally to generate sales_pipeline_model.pkl

COPY sales_pipeline_model.pkl ./

#CRITICAL: Copy your application code (features.py holds the shared feature engineering)

COPY main.py features.py ./

#Expose the port that FastAPI will run on

EXPOSE 9696

#Command to run the application using Uvicorn

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
