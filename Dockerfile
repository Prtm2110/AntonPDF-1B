FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy the processing script
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "llm4_to_json.py", "--all"]
