import os
import json

from xml.etree import ElementTree as ET

from utils.jmeter_utils import XMLBuilder, JmeterFunctions, JMeterElements, Attributes
from constants.constants import SimpleHTTPTestPlanConstants, CommonConstants


class SimpleHTTPPlan(object):
    def __init__(self, test_id, url, ):
        self.test_id = test_id
        self.url = url

    def default_request_info(self):
        return {'url': self.url,
                'contentType': 'application/json',
                'method': 'POST'
                # 'body': json.dumps(dict(data=data))
                }

    @staticmethod
    def default_header_info():
        return {'User-Agent': 'ApacheJMeter',
                'Content-Type': 'application/json',
                'azureml-model-deployment': '',
                'Authorization': ''}

    @staticmethod
    def default_threads_info():
        return {'num_threads': 1000,
                'ramp_time': 1}

    @staticmethod
    def test_plan_default_config():
        return dict(functional_mode='false',
                    tearDown_on_shutdown='true',
                    serialize_threadgroups='false')

    @staticmethod
    def thread_group_config():
        return dict(on_sample_error='continue',
                    num_thread='10',
                    ramp_time='1',
                    scheduler='false',
                    same_user_on_next_iteration='true',
                    continue_forever='false',
                    loops='1')

    @staticmethod
    def result_collector_config():
        return dict(time='true',
                    latency='true',
                    timestamp='true',
                    success='true',
                    label='true',
                    code='true',
                    message='true',
                    threadName='true',
                    dataType='true',
                    encoding='false',
                    assertions='true',
                    subresults='true',
                    responseData='false',
                    samplerData='false',
                    xml='true',
                    fieldNames='true',
                    responseHeaders='false',
                    requestHeaders='false',
                    responseDataOnError='false',
                    saveAssertionResultsFailureMessage='true',
                    assertionsResultsToSave='0',
                    bytes='true',
                    sentBytes='true',
                    url='true',
                    threadCounts='true',
                    idleTime='true',
                    connectTime='true')

    @staticmethod
    def http_sampler_config():
        return dict(postBodyRaw='true',
                    path='${url}',
                    method='${method}',
                    follow_redirects='true',
                    auto_redirects='false',
                    use_keepalive='true',
                    DO_MULTIPART_POST='false')

    def simple_http_test_plan(self, data, request_info=None, headers_info=None, threads_info=None):
        builder = XMLBuilder()
        elements = JMeterElements()

        if headers_info is None:
            headers_info = self.default_header_info()

        if threads_info is None:
            threads_info = self.default_threads_info()

        thread_group_config = self.thread_group_config()
        thread_group_config.update(threads_info)

        request_config = self.default_request_info()
        if request_info is not None:
            request_config.update(request_info)

        headers_config = self.default_header_info()
        if headers_info is not None:
            headers_config.update(headers_info)

        plan = builder.element('jmeterTestPlan', Attributes(version='1.2', properties='5.0', jmeter='5.5'))
        hash_tree_0 = elements.hash_tree(plan)

        fn = JmeterFunctions()
        fn.test_plan(hash_tree_0, **self.test_plan_default_config())
        hash_tree_1 = elements.hash_tree(hash_tree_0)
        fn.arguments(parent=hash_tree_1,
                     props=request_config)

        elements.hash_tree(hash_tree_1)
        fn.result_collector(parent=hash_tree_1,
                            config=self.result_collector_config(),
                            **{'error_logging': 'false'})
        elements.hash_tree(hash_tree_1)
        fn.thread_group(parent=hash_tree_1, **thread_group_config)
        hash_tree_2 = elements.hash_tree(hash_tree_1)
        fn.http_sampler_proxy(parent=hash_tree_2,
                              data=json.dumps(dict(data=data)),
                              **self.http_sampler_config())
        hash_tree_3 = elements.hash_tree(hash_tree_2)
        print(headers_config)
        fn.http_header_manager(parent=hash_tree_3, props=headers_config)
        elements.hash_tree(hash_tree_3)

        try:
            os.makedirs(CommonConstants.temp_file_base_path)
        except FileExistsError:
            pass

        temp_test_plan_path = os.path.join(CommonConstants.temp_file_base_path,
                                           SimpleHTTPTestPlanConstants.temp_file_name.format(self.test_id))
        tree = ET.ElementTree(plan)
        ET.indent(tree, space="  ", level=0)
        tree.write(temp_test_plan_path,
                   encoding="utf-8", xml_declaration=True)

        f = open(temp_test_plan_path.format(self.test_id), 'r')
        content = f.read()
        f.close()

        os.remove(temp_test_plan_path)

        test_plan = content.replace('> <', '><')
        return test_plan
