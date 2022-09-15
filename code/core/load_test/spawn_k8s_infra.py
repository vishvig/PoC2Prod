import time

from utils.k8s_utils import K8sUtils
from constants.constants import K8sJMeterConstants


class SpawnTestInfrastructureK8s(object):
    def __init__(self, test_id,
                 cpu_requests=None,
                 cpu_limit=None,
                 mem_requests=None,
                 mem_limit=None,
                 k8s_namespace=None,
                 cluster_name=None):
        self.test_id = test_id.replace('_', '-')
        self.k8s_client = K8sUtils(context=cluster_name)
        self.controller_name = f'{self.test_id}-controller'
        self.worker_name = f'{self.test_id}-worker'
        self.controller_labels = {'role': 'controller',
                                  'test-id': self.test_id}
        self.worker_labels = {'role': 'worker',
                              'test-id': self.test_id}
        if k8s_namespace is None:
            k8s_namespace = 'default'
        self.namespace = k8s_namespace
        self.cpu_requests = cpu_requests
        self.cpu_limit = cpu_limit
        self.mem_requests = mem_requests
        self.mem_limit = mem_limit
        self.redacted_name = self.test_id[:15]

    def create_session(self, engine_instances):
        controller_deployment = self.k8s_client.make_deployment(name=self.controller_name,
                                                                image_spec=K8sJMeterConstants.image,
                                                                port=5000,
                                                                working_dir='/opt',
                                                                labels=self.controller_labels,
                                                                launch_commands=["sh", "/opt/run.sh", "controller"],
                                                                redacted_name=self.redacted_name,
                                                                cpu_requests=self.cpu_requests,
                                                                cpu_limit=self.cpu_limit,
                                                                mem_requests=self.mem_requests,
                                                                mem_limit=self.mem_limit)
        self.k8s_client.create_namespaced_deployment(namespace=self.namespace,
                                                     body=controller_deployment)

        controller_service = self.k8s_client.make_service(name=self.controller_name,
                                                          labels=self.controller_labels,
                                                          port=80,
                                                          target_port=5000,
                                                          service_type='LoadBalancer')

        self.k8s_client.create_namespaced_service(namespace=self.namespace,
                                                  body=controller_service)

        if engine_instances > 0:
            workers = self.launch_workers(engine_instances=engine_instances)
        else:
            workers = list()

        load_balancer_ip = None
        server_url = None
        while load_balancer_ip is None:
            # print('Waiting for IP...')
            try:
                out = self.k8s_client.get_namespaced_service(namespace=self.namespace,
                                                             service_name=self.controller_name)
                load_balancer_ip = out.status.load_balancer.ingress[0].ip
                # load_balancer_port = out.spec.ports[0].node_port
                load_balancer_port = 80
                server_url = f'http://{load_balancer_ip}:{load_balancer_port}'
            except TypeError:
                pass
            time.sleep(1)
        print(f'Load testing session created. Server URL: {server_url}')
        return server_url, workers

    def launch_workers(self, engine_instances):
        worker_deployment = self.k8s_client.make_deployment(name=self.worker_name,
                                                            image_spec=K8sJMeterConstants.image,
                                                            port=1099,
                                                            working_dir='/opt',
                                                            labels=self.worker_labels,
                                                            launch_commands=["sh", "/opt/run.sh"],
                                                            redacted_name=self.redacted_name,
                                                            replicas=engine_instances,
                                                            cpu_requests=self.cpu_requests,
                                                            cpu_limit=self.cpu_limit,
                                                            mem_requests=self.mem_requests,
                                                            mem_limit=self.mem_limit
                                                            )
        self.k8s_client.create_namespaced_deployment(namespace=self.namespace,
                                                     body=worker_deployment)
        worker_service = self.k8s_client.make_service(name=self.worker_name,
                                                      labels=self.worker_labels,
                                                      port=1099,
                                                      target_port=1099,
                                                      service_type='ClusterIP')

        self.k8s_client.create_namespaced_service(namespace=self.namespace,
                                                  body=worker_service)

        worker_ips = None
        ip_initialized = False
        while not ip_initialized:
            worker_pods = self.k8s_client.list_namespaced_pod(namespace=self.namespace,
                                                              label_selector=f"role=worker,"
                                                                             f"test-id={self.test_id}"). \
                items
            if worker_ips is None:
                worker_ips = [False] * len(worker_pods)
            for i, pod in enumerate(worker_pods):
                pod_ip = pod.status.pod_ip
                worker_ips[i] = pod_ip
            if all(worker_ips):
                ip_initialized = True
            time.sleep(1)
        workers = [f"{i}:1099" for i in worker_ips]
        print(f'JMeter resources created! - {workers}')
        return workers

    def end_session(self, debug=False):
        if not debug:
            self.k8s_client.delete_namespaced_deployment(namespace=self.namespace,
                                                         name=self.controller_name),
        self.k8s_client.delete_namespaced_service(namespace=self.namespace,
                                                  name=self.controller_name)
        self.k8s_client.delete_namespaced_deployment(namespace=self.namespace,
                                                     name=self.worker_name)
        self.k8s_client.delete_namespaced_service(namespace=self.namespace,
                                                  name=self.worker_name)
        print('Session Over')

    def get_client(self):
        return self.k8s_client

    def create_or_update_model_resources(self, resources):
        instance_type = ''
        if resources is not None:
            print(f"Updating resources for configured model")
            requests_memory = resources['requests'].get('memory', None)
            requests_cpu = resources['requests'].get('cpu', None)
            requests_gpu = resources['requests'].get('gpu', None)
            limits_memory = resources['limits'].get('memory', None)
            limits_cpu = resources['limits'].get('cpu', None)
            limits_gpu = resources['limits'].get('gpu', None)

            if requests_memory is not None or requests_cpu is not None or requests_gpu is not None:
                instance_type += 'req'
                if requests_memory is not None:
                    instance_type += f'--mem-{requests_memory}'
                if requests_cpu is not None:
                    instance_type += f'--cpu-{requests_cpu}'
                if requests_gpu is not None:
                    instance_type += f'--gpu-{requests_gpu}'
            if limits_memory is not None or limits_cpu is not None or limits_gpu is not None:
                instance_type += '---lim'
                if limits_memory is not None:
                    instance_type += f'--mem-{limits_memory}'
                if limits_cpu is not None:
                    instance_type += f'--cpu-{limits_cpu}'
                if limits_gpu is not None:
                    instance_type += f'--gpu-{limits_gpu}'
            instance_type = instance_type.lower()
            if self.k8s_client.get_instance_type(name=instance_type) is None:
                self.k8s_client.create_instance_type(name=instance_type,
                                                     limits_cpu=limits_cpu,
                                                     limits_memory=limits_memory,
                                                     requests_cpu=requests_cpu,
                                                     requests_memory=requests_memory)
        return instance_type
