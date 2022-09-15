FROM openjdk:11

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

WORKDIR /opt
RUN wget https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-5.5.tgz
RUN tar xzvf apache-jmeter-5.5.tgz
RUN rm -f apache-jmeter-5.5.tgz
RUN ln -s apache-jmeter-5.5 jmeter
COPY jmeter.properties /opt/jmeter/bin/jmeter.properties

COPY agent.py /opt/agent.py
COPY run.sh /opt/run.sh
CMD sh /opt/run.sh
