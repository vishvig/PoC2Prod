import yaml

from kubernetes import config
import kubernetes.client
from kubernetes.client import (
    V1PodSpec,
    V1ObjectMeta, V1Service, V1ServiceSpec, V1ServicePort,
    V1Container, V1ContainerPort,
    V1Deployment, V1DeploymentSpec, V1DeploymentStatus,
    V1PodTemplateSpec, V1LabelSelector, V1ResourceRequirements
)

from kubernetes.client.rest import ApiException


class K8sUtils(object):
    def __init__(self, context=None):
        config.load_kube_config(context=context)
        self.client = kubernetes.client.ApiClient()

    def make_deployment(self, name,
                        image_spec,
                        port=None,
                        image_pull_policy="IfNotPresent",
                        working_dir=None,
                        labels=None,
                        annotations=None,
                        launch_commands=None,
                        redacted_name=None,
                        cpu_requests=None,
                        cpu_limit=None,
                        mem_requests=None,
                        mem_limit=None,
                        replicas=1):
        try:
            if labels is None:
                labels = dict()
            if annotations is None:
                annotations = dict()

            deployment = V1Deployment()
            deployment.kind = "Deployment"
            deployment.api_version = "apps/v1"
            deployment.metadata = V1ObjectMeta(
                name=name,
                labels=labels.copy(),
                annotations=annotations.copy(),
            )
            pod_spec = self.make_pod_spec(name=redacted_name,
                                          image_spec=image_spec,
                                          port=port,
                                          image_pull_policy=image_pull_policy,
                                          working_dir=working_dir,
                                          launch_commands=launch_commands,
                                          cpu_requests=cpu_requests,
                                          cpu_limit=cpu_limit,
                                          mem_requests=mem_requests,
                                          mem_limit=mem_limit)
            deployment.spec = V1DeploymentSpec(selector=V1LabelSelector(match_labels=labels.copy()),
                                               template=V1PodTemplateSpec(metadata=V1ObjectMeta(
                                                   name=name,
                                                   labels=labels.copy(),
                                                   annotations=annotations.copy()
                                               ),
                                                   spec=pod_spec),
                                               replicas=replicas)
            deployment.status = V1DeploymentStatus()
            return deployment
        except Exception as e:
            raise Exception(f"Faced a problem when creating the deployment template: {e}")

    @staticmethod
    def make_pod_spec(name,
                      image_spec,
                      port,
                      image_pull_policy="IfNotPresent",
                      working_dir=None,
                      cpu_requests=None,
                      cpu_limit=None,
                      mem_requests=None,
                      mem_limit=None,
                      launch_commands=None):
        try:
            if isinstance(port, str):
                port = int(port)

            spec = V1PodSpec(containers=[])
            spec.restart_policy = 'Always'

            container = V1Container(
                name=name,
                image=image_spec,
                working_dir=working_dir,
                ports=[
                    V1ContainerPort(name=name,
                                    container_port=port)],
                args=launch_commands,
                resources=V1ResourceRequirements(),
                image_pull_policy=image_pull_policy
            )
            container.resources.requests = {}
            if cpu_requests:
                container.resources.requests['cpu'] = cpu_requests
            if mem_requests:
                container.resources.requests['memory'] = mem_requests

            container.resources.limits = {}
            if cpu_limit:
                container.resources.limits['cpu'] = cpu_limit
            if mem_limit:
                container.resources.limits['memory'] = mem_limit

            spec.containers.append(container)
            return spec
        except Exception as e:
            raise Exception(f"Faced a problem when creating the pod spec template: {e}")

    @staticmethod
    def make_service(name,
                     labels,
                     port,
                     target_port,
                     service_type='ClusterIP'):

        meta = V1ObjectMeta(
            name=name,
            labels=labels
        )

        service = V1Service(
            kind='Service',
            metadata=meta,
            spec=V1ServiceSpec(
                type=service_type,
                selector=labels,
                ports=[V1ServicePort(port=port, target_port=target_port)]
            )
        )
        return service

    def apps_v1_api(self):
        return kubernetes.client.AppsV1Api(self.client)

    def core_v1_api(self):
        return kubernetes.client.CoreV1Api(self.client)

    def custom_object_api(self):
        return kubernetes.client.CustomObjectsApi(self.client)

    def create_namespaced_deployment(self, namespace, body):
        try:
            api_response = self.apps_v1_api().create_namespaced_deployment(namespace=namespace, body=body)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling AppsV1Api->create_namespaced_deployment: {e}")

    def delete_namespaced_deployment(self, namespace, name):
        try:
            api_response = self.apps_v1_api().delete_namespaced_deployment(name=name, namespace=namespace)
            return api_response
        except ApiException as e:
            if e.reason.lower() == 'not found':
                return True
            raise Exception(f"Exception when calling AppsV1Api->delete_namespaced_deployment: {e}")

    def get_namespaced_deployment(self, namespace, deployment_name):
        try:
            api_response = self.apps_v1_api().read_namespaced_deployment(namespace=namespace, name=deployment_name)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling AppsV1Api->get_namespaced_deployment: {e}")

    def list_namespaced_deployments(self, namespace):
        try:
            api_response = self.apps_v1_api().list_namespaced_deployment(namespace=namespace)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling AppsV1Api->list_namespaced_deployment: {e}")

    def create_namespaced_service(self, namespace, body):
        try:
            api_response = self.core_v1_api().create_namespaced_service(namespace=namespace, body=body)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling CoreV1Api->create_namespaced_service: {e}")

    def delete_namespaced_service(self, namespace, name):
        try:
            api_response = self.core_v1_api().delete_namespaced_service(name=name, namespace=namespace)
            return api_response
        except ApiException as e:
            if e.reason.lower() == 'not found':
                return True
            raise Exception(f"Exception when calling CoreV1Api->delete_namespaced_service: {e}")

    def get_namespaced_service(self, namespace, service_name):
        try:
            api_response = self.core_v1_api().read_namespaced_service(namespace=namespace, name=service_name)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling CoreV1Api->read_namespaced_service: {e}")

    def list_namespaced_service(self, namespace):
        try:
            api_response = self.core_v1_api().list_namespaced_service(namespace=namespace)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling CoreV1Api->list_namespaced_service: {e}")

    def list_namespaced_pod(self, namespace, **kwargs):
        try:
            api_response = self.core_v1_api().list_namespaced_pod(namespace=namespace, **kwargs)
            return api_response
        except ApiException as e:
            raise Exception(f"Exception when calling CoreV1Api->list_namespaced_pods: {e}")

    def get_pod_metrics(self, namespace, pod_name):
        plural = f"namespaces/{namespace}/pods/{pod_name}"
        try:
            api_response = self.custom_object_api().list_cluster_custom_object("metrics.k8s.io",
                                                                               "v1beta1",
                                                                               plural=plural
                                                                               )
            return api_response
        except ApiException as e:
            if e.status == 404:
                # print(f"Error: Plural - {plural}")
                raise ConnectionError(e)
            else:
                raise Exception(f"Exception when calling CustomObjectAPI->list_namespaced_custom_object: {e}")

    def create_instance_type(self, name,
                             limits_cpu=None,
                             limits_memory=None,
                             limits_gpu=None,
                             requests_cpu=None,
                             requests_memory=None,
                             requests_gpu=None):
        try:
            body = dict(apiVersion='amlarc.azureml.com/v1alpha1',
                        kind='InstanceType',
                        metadata=dict(name=name),
                        spec=dict(
                            resources=dict(
                                limits={
                                    "cpu": limits_cpu,
                                    "memory": limits_memory,
                                    "nvidia.com/gpu": limits_gpu
                                },
                                requests={
                                    "cpu": requests_cpu,
                                    "memory": requests_memory,
                                    "nvidia.com/gpu": requests_gpu
                                }
                            )
                        )
                        )

            self.custom_object_api().create_cluster_custom_object("amlarc.azureml.com",
                                                                  "v1alpha1",
                                                                  "instancetypes",
                                                                  body)
            return True
        except ApiException as e:
            if e.status == 409:
                return True
            else:
                raise Exception(f"Exception when calling CustomObjectAPI->create_instance_type: {e}")

    def get_instance_type(self, name):
        try:
            return self.custom_object_api().get_cluster_custom_object("amlarc.azureml.com",
                                                                      "v1alpha1",
                                                                      "instancetypes",
                                                                      name)
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                raise Exception(f"Exception when calling CustomObjectAPI->get_instance_type: {e}")

    @staticmethod
    def cpu_float(cpu):
        """
        Convert all cpu values seen in K8s to milli-cpus.
        Milli-CPUs is taken because that is the standard of cpu requests and limits in K8s
        :param cpu: The CPU value to be converted
        :return: Milli-CPU float value
        """
        if 'u' in cpu:
            # Convert micro-cpu units to milli-cpus
            return float(cpu.rstrip('u')) / 10 ** 3
        elif 'n' in cpu:
            # Convert nano-cpu units to milli-cpus
            return float(cpu.rstrip('n')) / 10 ** 6
        else:
            return float(cpu.rstrip('m'))

    @staticmethod
    def mem_float(memory):
        """
        Convert all memory values seen in K8s to Kilobytes.
        Kilobytes is taken because that is the standard of memory requests and limits in K8s
        :param memory: The memory value to be converted
        :return: Memory in Ki
        """
        if 'M' in memory:
            # Convert Megabytes to kilobytes
            return float(memory.rstrip('Mi').rstrip('M')) * (2 ** 10)
        else:
            return float(memory.rstrip('Ki'))

    def get_node_allocatable_resources(self, name):
        try:
            cpu = self.cpu_float(self.core_v1_api().read_node(name=name).status.allocatable['cpu'])
            memory = self.mem_float(self.core_v1_api().read_node(name=name).status.allocatable['memory'])
            return dict(cpu=cpu, memory=memory)
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                raise Exception(f"Exception when calling CoreV1Api->read_node: {e}")

    def get_cluster_allocatable_resources(self):
        try:
            cpu = 0
            memory = 0
            for node in self.core_v1_api().list_node().items:
                cpu += self.cpu_float(node.status.allocatable['cpu'])
                memory += self.mem_float(node.status.allocatable['memory'])
            return dict(cpu=cpu, memory=memory)
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                raise Exception(f"Exception when calling CoreV1Api->read_node: {e}")
