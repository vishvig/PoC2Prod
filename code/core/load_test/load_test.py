import json
import os
import time
import base64
import requests
import zipfile
import pandas as pd
import warnings
import traceback

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import DefaultScaleSettings

from core.load_test.spawn_k8s_infra import SpawnTestInfrastructureK8s
from core.load_test.simple_http_plan import SimpleHTTPPlan
from utils.common_utils import background


class ModelLoadTest(object):
    def __init__(self, model_info, run_id=0):
        self.ml_client = None
        self.model_details = model_info
        self.time_now = int(time.time())
        self.test_id = f'test_{self.time_now}'
        self.run_id = run_id
        self.load_test_k8s = None
        self.az_ml_k8s_client = None
        self.sti = None
        self.server_url = None
        self.workers = None

        self.endpoint = None
        self.compute = None
        self.deployment = None

        self.monitor_resources = False
        self.resource_metrics = dict(time=list(),
                                     endpoint_name=list(),
                                     pod_name=list(),
                                     node_name=list(),
                                     cpu=list(),
                                     memory=list()
                                     )
        self.load_test_resources = dict()

    def get_model_details(self):
        """
        Fetch the information of the ML model stored on Azure
        """
        # Initializing the Azure ML client
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            self.model_details['subscription_id'],
            self.model_details['resource_grp'],
            self.model_details['ml_workspace']
        )

        # Fetching details of compute from azure ML client
        self.compute = self.ml_client.compute.get(name=self.model_details['compute_name'])

        # Fetching details of endpoint from azure ML client
        self.endpoint = self.ml_client.online_endpoints.get(name=self.model_details['endpoint_name'])

        # Fetching details of deployment from azure ML client
        self.deployment = self.ml_client.online_deployments.get(name=self.model_details['deployment_name'],
                                                                endpoint_name=self.model_details['endpoint_name'])

        # Storing the required state values of the deployed ML model in a dict
        self.model_details['url'] = self.endpoint.scoring_uri
        self.model_details['traffic'] = self.endpoint.traffic
        self.model_details['namespace'] = self.compute.namespace
        self.model_details['resource_id'] = self.compute.resource_id
        self.model_details['auth_token'] = self.ml_client.online_endpoints.list_keys(
            name=self.model_details['endpoint_name']).primary_key

    def start(self, engine_instances, az_ml_k8s_config, load_test_k8s_config):
        self.load_test_resources = load_test_k8s_config.get('resources', dict())
        cpu_requests = self.load_test_resources.get("cpu", dict()).get("requests", "1")
        cpu_limit = self.load_test_resources.get("cpu", dict()).get("limit", "1")
        mem_requests = self.load_test_resources.get("memory", dict()).get("requests", "1Gi")
        mem_limit = self.load_test_resources.get("memory", dict()).get("limit", "1Gi")
        self.load_test_k8s = SpawnTestInfrastructureK8s(test_id=self.test_id,
                                                        k8s_namespace=load_test_k8s_config['namespace'],
                                                        cluster_name=load_test_k8s_config["cluster_name"],
                                                        cpu_requests=cpu_requests,
                                                        cpu_limit=cpu_limit,
                                                        mem_requests=mem_requests,
                                                        mem_limit=mem_limit)
        self.sti = SpawnTestInfrastructureK8s(test_id=self.test_id,
                                              cluster_name=az_ml_k8s_config["cluster_name"])
        self.az_ml_k8s_client = self.sti.get_client()
        self.server_url, self.workers = self.load_test_k8s.create_session(engine_instances=engine_instances)

    def end(self, debug=False):
        self.load_test_k8s.end_session(debug=debug)

    @background
    def monitor_model_resources(self, deployment_name):
        """
        Asynchronous thread to monitor and gather the pod CPU and memory usage
        information of the deployed ML model on AKS
        :param deployment_name: The name of the
        :return:
        """
        # Setting flag to True to persist resource monitoring until set to False
        self.monitor_resources = True

        # Initializing ML endpoint details and K8s client
        endpoint_name = self.model_details['endpoint_name']

        # Setting a warning flag to stop code from over-displaying the errors/warning seen from K8s
        warn = None

        print(f"Started resource monitoring for deployment: {deployment_name}")
        while self.monitor_resources:
            try:
                time_now = int(time.time())

                # Fetching pods related to the ML deployment
                pods = self.az_ml_k8s_client.list_namespaced_pod(
                    namespace=self.model_details['namespace'],
                    label_selector=f"ml.azure.com/endpoint-name={endpoint_name},"
                                   f"isazuremlapp=true,"
                                   f"ml.azure.com/deployment-name={deployment_name}"). \
                    items

                # Setting a warning flag for every pod seen in deployment
                if warn is None:
                    warn = [False] * len(pods)

                for i, pod in enumerate(pods):
                    pod_name = pod.metadata.name
                    node_name = pod.spec.node_name
                    try:
                        # Calling K8s metrics server API to collect CPU & memory usage
                        out = self.az_ml_k8s_client.get_pod_metrics(self.model_details['namespace'], pod_name)
                    except ConnectionError as e:
                        # Displaying a warning if the code faced any issue while gathering metrics from K8s
                        if warn[i] is False:
                            warnings.warn(f"Saw an issue when monitoring resources: {e}", RuntimeWarning)
                            warn[i] = True
                        continue

                    # Collection and storing the CPU and memory usage of only the inference-server container
                    for container in out['containers']:
                        if container['name'] == 'inference-server':
                            cpu = container['usage']['cpu']
                            memory = container['usage']['memory']
                            self.resource_metrics['time'].append(time_now)
                            self.resource_metrics['endpoint_name'].append(endpoint_name)
                            self.resource_metrics['pod_name'].append(pod_name)
                            self.resource_metrics['cpu'].append(cpu)
                            self.resource_metrics['memory'].append(memory)
                            self.resource_metrics['node_name'].append(node_name)
                            break
            except Exception as e:
                print(e)
                continue
            time.sleep(1)
        print(f"Ended resource monitoring")

    def report_results(self, test_id, test_results, run_path):
        """
        Calculate and write node level and cluster level metrics for a specific test run
        :param test_id: The test id of the current test run
        :param test_results: Results from JMeter in a zip file (bytes-like object)
        :param run_path: The base path to store the test run results
        :return:
        """
        try:
            # Creating the metrics folder in the test run path if not available
            metrics_path = os.path.join(run_path, 'metrics')
            try:
                os.makedirs(metrics_path)
            except OSError:
                pass
            metrics_df = pd.DataFrame(self.resource_metrics)

            # Writing all the metrics to results folder
            resource_metrics_file = os.path.join(metrics_path, f'raw_metrics.csv')
            metrics_df.to_csv(resource_metrics_file)

            # Unzipping JMeter test results to results folder
            zip_file = os.path.join(run_path, f'{test_id}.zip')
            pid_file = os.path.join(run_path, f'run.pid')
            with open(zip_file, 'wb') as fd:
                for chunk in test_results.iter_content(chunk_size=128):
                    fd.write(chunk)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(run_path)
            os.remove(zip_file)
            os.remove(pid_file)

            # Converting raw cpu and memory values to standard float values
            metrics_df['cpu_float'] = metrics_df['cpu'].apply(self.az_ml_k8s_client.cpu_float)
            metrics_df['mem_float'] = metrics_df['memory'].apply(self.az_ml_k8s_client.mem_float)

            # Fetch the allocatable capacity of k8s cluster and every node in it
            cluster_resources = self.az_ml_k8s_client.get_cluster_allocatable_resources()
            node_resources = dict()
            unique_nodes = list(metrics_df['node_name'].unique())
            for node in unique_nodes:
                node_resources[node] = self.az_ml_k8s_client.get_node_allocatable_resources(name=node)

            # Calculate node utilization metrics and write to csv file
            self.node_utilization_metrics(metrics_df=metrics_df,
                                          metrics_path=metrics_path,
                                          node_resources=node_resources)

            # Calculate cluster utilization metrics and write to csv file
            self.cluster_utilization_metrics(metrics_df=metrics_df,
                                             metrics_path=metrics_path,
                                             cluster_resources=cluster_resources)
            print(f'Job report details present in "{run_path}"')
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"Job faced a problem when reporting the results: {e}")

    @staticmethod
    def node_utilization_metrics(metrics_df, metrics_path, node_resources):
        """
        Calculate node utilization metrics and write to csv file
        :param metrics_df: The raw metrics dataframe
        :param metrics_path: The path to store the calculated node utilization metrics
        :param node_resources: The max. allocatable resources on each node on the k8s cluster
        :return:
        """
        try:
            # Calculating node utilization metrics
            node_utilization = metrics_df.copy(deep=True)

            # Aggregating by sum the cpu and memory raw values (group by each node)
            to_aggregate = {'cpu_float': 'sum', 'mem_float': 'sum'}
            node_utilization = node_utilization.groupby(['time', 'endpoint_name', 'node_name'], as_index=False). \
                agg(to_aggregate)

            # Calculating the percentage usage of node cpu by deployment against the max. allocatable capacity
            node_utilization["node_cpu_utilization%"] = node_utilization. \
                apply(lambda x: (x['cpu_float'] / node_resources[x['node_name']]['cpu']) * 100, axis=1)

            # Calculating the percentage usage of node memory by deployment against the max. allocatable capacity
            node_utilization["node_mem_utilization%"] = node_utilization. \
                apply(lambda x: (x['mem_float'] / node_resources[x['node_name']]['memory']) * 100, axis=1)

            # Cleaning up dataframe to include only necessary columns
            node_utilization.drop(['cpu_float', 'mem_float'], axis=1)

            # Writing the node utilization metrics to csv file
            node_utilization_file = os.path.join(metrics_path, f'node_utilization.csv')
            node_utilization.to_csv(node_utilization_file)
        except Exception as e:
            warnings.warn(f"Unable to calculate node utilization metrics: {e}")
            return False

    @staticmethod
    def cluster_utilization_metrics(metrics_df, metrics_path, cluster_resources):
        try:
            # Calculating cluster utilization metrics
            cluster_cpu = cluster_resources['cpu']
            cluster_memory = cluster_resources['memory']
            cluster_utilization = metrics_df.copy(deep=True)

            to_aggregate = {'cpu_float': 'sum', 'mem_float': 'sum'}

            cluster_utilization = cluster_utilization.groupby(['time', 'endpoint_name'], as_index=False). \
                agg(to_aggregate)
            cluster_utilization["cluster_cpu_utilization%"] = (cluster_utilization['cpu_float'] / cluster_cpu) * 100
            cluster_utilization["cluster_mem_utilization%"] = (cluster_utilization['mem_float'] / cluster_memory) * 100
            cluster_utilization.drop(['cpu_float', 'mem_float'], axis=1)
            cluster_utilization_file = os.path.join(metrics_path, f'cluster_utilization.csv')
            cluster_utilization.to_csv(cluster_utilization_file)
        except Exception as e:
            warnings.warn(f"Unable to calculate cluster utilization metrics: {e}")
            return False

    def run_test(self, data, results_path,
                 request_info=None,
                 headers_info=None,
                 threads_info=None,
                 resources=None,
                 instance_count=1):
        deployment_name = None
        try:
            test_id = f"{self.test_id}_{self.run_id}"
            self.get_model_details()

            run_path = os.path.join(results_path, self.test_id, str(self.run_id))
            try:
                os.makedirs(run_path)
            except OSError:
                pass

            # Writing test run configurations into a file
            test_run_details = dict(threads_info=threads_info,
                                    resources=resources,
                                    instance_count=instance_count)
            f = open(os.path.join(run_path, 'test_run_details.json'), 'w+')
            f.write(json.dumps(test_run_details))
            f.close()

            instance_type = self.sti.create_or_update_model_resources(resources=resources)

            new_deployment = self.deployment
            deployment_name = test_id.replace('_', '-')
            new_deployment.name = deployment_name
            new_deployment.instance_count = instance_count
            new_deployment.scale_settings = DefaultScaleSettings()
            if instance_type != '':
                new_deployment.instance_type = instance_type
            print(new_deployment)
            print("Creating ML deployment with requested resources")
            self.ml_client.begin_create_or_update(new_deployment)

            deployment_status = False
            while not deployment_status:
                deployment = self.ml_client.online_deployments.get(name=deployment_name,
                                                                   endpoint_name=self.model_details['endpoint_name'])
                if deployment.provisioning_state == 'Succeeded':
                    deployment_status = True
                else:
                    time.sleep(1)

            self.endpoint.traffic = {deployment_name: 100}
            self.ml_client.begin_create_or_update(self.endpoint)

            endpoint_status = False
            while not endpoint_status:
                endpoint = self.ml_client.online_endpoints.get(name=self.model_details['endpoint_name'])
                if endpoint.provisioning_state == 'Succeeded':
                    endpoint_status = True
                else:
                    time.sleep(1)

            if headers_info is None:
                headers_info = {'Authorization': f'Bearer {self.model_details["auth_token"]}',
                                'azureml-model-deployment': deployment_name}
            else:
                headers_info.update({'Authorization': f'Bearer {self.model_details["auth_token"]}',
                                     'azureml-model-deployment': deployment_name})
            time.sleep(5)
            self.monitor_model_resources(deployment_name=deployment_name)
            time.sleep(5)
            test_plan = SimpleHTTPPlan(test_id=test_id, url=self.model_details['url']). \
                simple_http_test_plan(data=data,
                                      request_info=request_info,
                                      headers_info=headers_info,
                                      threads_info=threads_info)
            # xms = self.load_test_resources.get("memory", dict()).get("requests", '1').rstrip('Gi')
            xms = '1'
            xmx = self.load_test_resources.get("memory", dict()).get("requests", '1').rstrip('Gi')
            request_json = json.dumps(dict(test_id=test_id,
                                           hosts=self.workers,
                                           test_plan=base64.b64encode(test_plan.encode()).decode(),
                                           memory=dict(xms=xms,
                                                       xmx=xmx)))
            print(f"Submitting test run")
            res = requests.post(url=f'{self.server_url}/run_test',
                                data=request_json,
                                timeout=300)
            if res.status_code != 200:
                raise Exception(f'Run was unable to submit the load test job to JMeter')
            print(f'Load test run {test_id} submitted: {res}')

            status = True
            while status:
                res = requests.post(url=f'{self.server_url}/get_test_status',
                                    data=request_json)
                status = res.json()['status']
                # print(f'Job status now: {status}')
                time.sleep(1)
            print(f'Load test run {test_id} done')
            results = requests.post(url=f'{self.server_url}/download_test_report',
                                    data=request_json, stream=True)
            time.sleep(15)
            self.monitor_resources = False
            if results.status_code != 200:
                raise Exception(f"Job was unable to fetch the results from JMeter load test run")
            print(f"Preparing resource metrics for load test {test_id}")
            self.report_results(test_id=test_id,
                                test_results=results,
                                run_path=run_path)
            self.endpoint.traffic = self.model_details['traffic']
            self.ml_client.begin_create_or_update(self.endpoint)
            self.ml_client.online_deployments.delete(name=deployment_name,
                                                     endpoint_name=self.model_details['endpoint_name'])
            self.run_id += 1
            print(f"Reverted configured ML endpoint to it's original state")
        except KeyboardInterrupt:
            print(f"Destroying all spawned resources and reverting to original configurations")
            self.endpoint.traffic = self.model_details['traffic']
            self.ml_client.begin_create_or_update(self.endpoint)
            if deployment_name is not None:
                self.ml_client.online_deployments.delete(name=deployment_name,
                                                         endpoint_name=self.model_details['endpoint_name'])
            self.end()
            print(f"Session ended")
        except Exception as e:
            print(f"Destroying all spawned resources and reverting to original configurations: {e}")
            self.endpoint.traffic = self.model_details['traffic']
            self.ml_client.begin_create_or_update(self.endpoint)
            if deployment_name is not None:
                self.ml_client.online_deployments.delete(name=deployment_name,
                                                         endpoint_name=self.model_details['endpoint_name'])
            self.end(debug=True)
            print(f"Session ended")
