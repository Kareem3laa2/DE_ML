FROM bitnami/spark:3.1.1

USER root

# Install system & Python dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip openjdk-11-jdk curl && \
    pip3 install --upgrade pip && \
    pip3 install numpy pandas scikit-learn pyspark findspark && \
    rm -rf /var/lib/apt/lists/*

# Add the PostgreSQL JDBC driver
COPY jars/postgresql-42.7.7.jar /opt/bitnami/spark/jars/

USER 1001
