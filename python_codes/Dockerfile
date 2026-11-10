#Use a slim Python image for a smaller container size

FROM python:3.13.9-slim

#Set working directory

WORKDIR /app

#Copy the pipfile install dependencies

COPY Pipfile Pipfile.lock ./

#FIX: Install pipenv globally in the container first, as it's not included in the slim image.

RUN pip install pipenv

#Now install project dependencies using pipenv

RUN pipenv install --system --deploy

#Copy model artifacts and application code
#NOTE: You must run model_trainer.py locally to generate sales_pipeline_model.pkl

COPY sales_pipeline_model.pkl ./

#CRITICAL: Copy your application code

COPY main.py ./

#Expose the port that FastAPI will run on

EXPOSE 9696

#Command to run the application using Uvicorn
#Start uvicorn

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]