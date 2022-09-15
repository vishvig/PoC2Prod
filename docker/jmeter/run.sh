#!/bin/bash

if [ "$1" = "controller" ]
then
  python3 agent.py
else
  /opt/jmeter/bin/jmeter-server
fi
