import os
import base64
import psutil
import zipfile
import subprocess
import uvicorn
import traceback

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel

# app = Flask(__name__)
app = FastAPI(title="PoC2Prod Agent",
              version="0.1",
              description="This is the API service layer for PoC2Prod agent API")


class BaseRequest(BaseModel):
    test_id: str


class RunTestRequest(BaseRequest):
    hosts: List = list()
    test_plan: str
    memory: Optional[Dict]


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def execute_subprocess(command, env=None):
        if env is not None:
            env.update(os.environ)
        return subprocess.Popen(command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env)

    @staticmethod
    def write_to_file(filename, content):
        f = open(filename, 'w+')
        f.write(str(content))
        f.close()
        return True

    @staticmethod
    def read_from_file(filename):
        f = open(filename, 'r')
        content = f.read()
        f.close()
        return content

    @staticmethod
    def zip(src, dst):
        zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
        abs_src = os.path.abspath(src)
        for dirname, subdirs, files in os.walk(src):
            for filename in files:
                absname = os.path.abspath(os.path.join(dirname, filename))
                arcname = absname[len(abs_src) + 1:]
                zf.write(absname, arcname)
        zf.close()
        return True


async def submit_command(command, pid_file, env=None):
    utils_obj = Utils()
    proc = utils_obj.execute_subprocess(command=command, env=env)
    stdout, stderr = proc.communicate()
    process_id = proc.pid
    utils_obj.write_to_file(pid_file, process_id)
    # psutil.pid_exists(pid=process_id)


@app.post('/run_test')
async def run_test(request_data: RunTestRequest):
    try:
        utils_obj = Utils()
        test_id = request_data.test_id
        if len(request_data.hosts) != 0:
            hosts = ','.join(request_data.hosts)
        else:
            hosts = None
        test_plan = request_data.test_plan
        xms = request_data.memory.get('xms', '1')
        xmx = request_data.memory.get('xmx', '1')

        test_dir = os.path.join('/opt', 'tests', test_id)
        out_dir = os.path.join(test_dir, f'out')

        test_plan_script_file = os.path.join(test_dir, f'{test_id}.jmx')
        results_file = os.path.join(out_dir, 'results.csv')
        log_file = os.path.join(test_dir, f'{test_id}.log')
        pid_file = os.path.join(test_dir, f'run.pid')

        os.makedirs(test_dir)
        os.makedirs(out_dir)

        test_plan = base64.b64decode(test_plan).decode("utf-8")
        utils_obj.write_to_file(test_plan_script_file, test_plan)
        if hosts is not None:
            command = f'nohup bash -c "JVM_ARGS=\\\"-Xms{xms}g -Xmx{xmx}g\\\" && export JVM_ARGS && ' \
                      f'/opt/jmeter/bin/jmeter -n -t {test_plan_script_file} -p /opt/jmeter/bin/jmeter.properties ' \
                      f'-l {results_file} -j {log_file} -R {hosts} -e -o {out_dir}" > nohup.out 2>&1 &'
        else:
            command = f'nohup bash -c "JVM_ARGS=\\\"-Xms{xms}g -Xmx{xmx}g\\\" && export JVM_ARGS && ' \
                      f'/opt/jmeter/bin/jmeter -n -t {test_plan_script_file} -p /opt/jmeter/bin/jmeter.properties ' \
                      f'-l {results_file} -j {log_file} -e -o {out_dir}" > nohup.out 2>&1 &'
        print(command)
        await submit_command(command, pid_file, env={"JVM_ARGS": f'-Xms{xms}g -Xmx{xmx}g'})
        return JSONResponse(content=True, status_code=200)
    except Exception as e:
        print(f"Error when submitting test run: {e}")
        traceback.print_exc()
        return JSONResponse(content=e, status_code=500)


@app.post('/get_test_status')
def get_test_status(request_data: BaseRequest):
    try:
        utils_obj = Utils()
        test_id = request_data.test_id
        test_dir = os.path.join('/opt', 'tests', test_id)
        # pid_file = os.path.join(test_dir, f'run.pid')
        # process_id = int(utils_obj.read_from_file(pid_file))
        command = "ps -ef | grep jmeter | grep -v grep | grep -v java | awk '{print $2}'"
        p = utils_obj.execute_subprocess(command=command)
        stdout, stderr = p.communicate()
        pids = stdout.decode().split('\n')
        pids_status = [False] * len(pids)
        for i, pid in enumerate(pids):
            if pid != '':
                pids_status[i] = psutil.pid_exists(pid=int(pid))
        status = any(pids_status)
        return JSONResponse(content={'status': status, 'test_id': test_id}, status_code=200)
    except Exception as e:
        print(f"Error when fetching status of test run: {e}")
        traceback.print_exc()
        return JSONResponse(content={'status': False}, status_code=500)


@app.post('/download_test_report')
def download_test_report(request_data: BaseRequest):
    try:
        utils_obj = Utils()
        # data = json.loads(request.data)
        test_id = request_data.test_id
        test_dir = os.path.join('/opt', 'tests', test_id)
        zip_file_name = os.path.join('/opt', 'tests', f'{test_id}.zip')

        utils_obj.zip(test_dir, zip_file_name)
        return FileResponse(zip_file_name)
    except Exception as e:
        print(f"Error when downloading report: {e}")
        traceback.print_exc()
        return JSONResponse(content={'status': False}, status_code=500)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
