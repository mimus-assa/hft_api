# Build stage
FROM python:3.8-alpine as build

WORKDIR /app

COPY requirements.txt .

RUN apk add --no-cache --virtual .build-deps gcc libc-dev make \
    && pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps gcc libc-dev make

COPY . .

# Final stage
FROM python:3.8-alpine

WORKDIR /app

COPY --from=build /app .

EXPOSE 5000

CMD ["python", "app.py"]
