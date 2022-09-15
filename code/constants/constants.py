import os
from typing import Dict, Any


class CommonConstants:
    temp_file_base_path = os.path.join('tmp')


class K8sJMeterConstants:
    image = 'vrmptreg.azurecr.io/jmeter:1.15'
    controller_name: Any = str
    worker_name: Any = str
    controller_labels: Dict = dict
    worker_labels: Dict = dict
    namespace: Any = str


class SimpleHTTPTestPlanConstants:
    temp_file_name = "{}.jmx"
