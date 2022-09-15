from xml.etree import ElementTree as ET


class XMLBuilder(object):

    @staticmethod
    def element(tag, attributes=None, text=None):
        if attributes is None:
            attributes = Attributes()
        e = ET.Element(tag, attributes.__attrs__())
        e.text = text
        return e

    @staticmethod
    def sub_element(parent, tag, attributes=None, text=None):
        if attributes is None:
            attributes = Attributes()
        e = ET.SubElement(parent, tag, attributes.__attrs__())
        e.text = text
        return e


class Attributes:
    def __init__(self,
                 name=None,
                 element_type=None,
                 gui_class=None,
                 test_class=None,
                 test_name=None,
                 enabled=None,
                 version=None,
                 properties=None,
                 jmeter=None,
                 **kwargs):
        self.name = name
        self.elementType = element_type
        self.guiclass = gui_class
        self.testclass = test_class
        self.testname = test_name
        self.enabled = enabled
        self.version = version
        self.properties = properties
        self.jmeter = jmeter

        self.kwargs = kwargs

    def __attrs__(self):
        attrs = {k: v for k, v in vars(self).items() if v is not None and k not in ['kwargs']}
        attrs.update({k: v for k, v in self.kwargs.items() if v is not None})
        return attrs


class JMeterElements(object):
    def __init__(self):
        self.builder = XMLBuilder()

    def string_prop(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'stringProp', attributes, text)

    def bool_prop(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'boolProp', attributes, text)

    def element_prop(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'elementProp', attributes, text)

    def collection_prop(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'collectionProp', attributes, text)

    def obj_prop(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'objProp', attributes, text)

    def thread_group(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'ThreadGroup', attributes, text)

    def hash_tree(self, parent, attributes=None, text=None):
        return self.builder.sub_element(parent, 'hashTree', attributes, text)

    def http_sampler_proxy(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'HTTPSamplerProxy', attributes, text)

    def http_header_manager(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'HeaderManager', attributes, text)

    def test_plan(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'TestPlan', attributes, text)

    def arguments(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'Arguments', attributes, text)

    def result_collector(self, parent, attributes=None, text=' '):
        return self.builder.sub_element(parent, 'ResultCollector', attributes, text)

    def custom_element(self, parent, tag, attributes=None, text=' '):
        return self.builder.sub_element(parent, tag, attributes, text)


class JMeterTreeBuilder(object):
    def __init__(self,
                 parent,
                 name=None,
                 element_type=None,
                 gui_class=None,
                 class_name=None,
                 test_name=None,
                 prop_class_name=None,
                 enabled=None):
        self.builder = XMLBuilder()
        self.elm = JMeterElements()

        self.parent = parent
        self.name = name
        self.element_type = element_type
        self.gui_class = gui_class
        self.class_name = class_name
        self.test_name = test_name
        self.enabled = enabled
        self.prop_class_name = prop_class_name

    def template(self, func, props, **kwargs):
        tree = getattr(self.elm, func)(self.parent, Attributes(name=self.name,
                                                               gui_class=self.gui_class,
                                                               test_class=self.class_name,
                                                               test_name=self.test_name,
                                                               element_type=self.element_type,
                                                               enabled=self.enabled))
        for k, v in props.items():
            getattr(self.elm, f'{v}_prop')(tree,
                                           Attributes(name=f'{self.prop_class_name}.{k}'),
                                           text=str(kwargs.get(k, ' '))
                                           )
        return tree


class JmeterFunctions(object):
    def __init__(self):
        self.elm = JMeterElements()

    @staticmethod
    def test_plan(parent, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="TestPlanGui",
                               class_name='TestPlan',
                               test_name="Test Plan",
                               prop_class_name='TestPlan',
                               enabled='true')

        props = {'comments': 'string',
                 'functional_mode': 'bool',
                 'tearDown_on_shutdown': 'bool',
                 'serialize_threadgroups': 'bool',
                 'user_define_classpath': 'string'}
        tree = fb.template(func='test_plan', props=props, **kwargs)

        ep = JMeterTreeBuilder(parent=tree,
                               name='TestPlan.user_defined_variables',
                               element_type='Arguments',
                               gui_class='ArgumentsPanel',
                               class_name='Arguments',
                               test_name='User Defined Variables',
                               prop_class_name='Arguments',
                               enabled='true')

        ep_props = {'arguments': 'collection'}

        ep.template(func='element_prop', props=ep_props, **kwargs)

    @staticmethod
    def thread_group(parent, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="ThreadGroupGui",
                               class_name='ThreadGroup',
                               test_name="Thread Group",
                               prop_class_name='ThreadGroup',
                               enabled='true')

        props = {'on_sample_error': 'string',
                 'num_threads': 'string',
                 'ramp_time': 'string',
                 'scheduler': 'bool',
                 'duration': 'string',
                 'delay': 'string',
                 'same_user_on_next_iteration': 'bool'}
        tree = fb.template(func='thread_group', props=props, **kwargs)

        ep = JMeterTreeBuilder(parent=tree,
                               name='ThreadGroup.main_controller',
                               element_type='LoopController',
                               gui_class='LoopControlPanel',
                               class_name='LoopController',
                               test_name='Loop Controller',
                               prop_class_name='LoopController',
                               enabled='true')

        ep_props = {'continue_forever': 'bool',
                    'loops': 'string'}

        ep.template(func='element_prop', props=ep_props, **kwargs)

    def http_sampler_proxy(self, parent, data, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="HttpTestSampleGui",
                               class_name='HTTPSamplerProxy',
                               test_name="HTTP Request",
                               prop_class_name='HTTPSampler',
                               enabled='true')

        props = {'postBodyRaw': 'bool',
                 'domain': 'string',
                 'port': 'string',
                 'protocol': 'string',
                 'contentEncoding': 'string',
                 'path': 'string',
                 'method': 'string',
                 'follow_redirects': 'bool',
                 'auto_redirects': 'bool',
                 'use_keepalive': 'bool',
                 'DO_MULTIPART_POST': 'bool',
                 'embedded_url_re': 'string',
                 'connect_timeout': 'string',
                 'response_timeout': 'string'}
        tree = fb.template(func='http_sampler_proxy', props=props, **kwargs)
        ep = self.elm.element_prop(parent=tree, attributes=Attributes(name='HTTPsampler.Arguments',
                                                                      element_type="Arguments"))
        cp = self.elm.collection_prop(parent=ep, attributes=Attributes(name="Arguments.arguments"))
        ep1 = self.elm.element_prop(parent=cp, attributes=Attributes(name='',
                                                                     element_type="HTTPArgument"))
        self.elm.bool_prop(parent=ep1, attributes=Attributes(name="HTTPArgument.always_encode"), text='false')
        self.elm.string_prop(parent=ep1, attributes=Attributes(name='Argument.value'), text=data)
        self.elm.string_prop(parent=ep1, attributes=Attributes(name='Argument.metadata'), text='=')

    def http_header_manager(self, parent, props, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="HeaderPanel",
                               class_name='HeaderManager',
                               test_name="HTTP Header Manager",
                               prop_class_name='HeaderManager',
                               enabled='true')
        tree = fb.template(func='http_header_manager', props={}, **kwargs)
        cp = self.elm.collection_prop(parent=tree, attributes=Attributes(name='HeaderManager.headers'))
        for k, v in props.items():
            eb = JMeterTreeBuilder(parent=cp,
                                   name='',
                                   element_type='Header',
                                   prop_class_name='Header')
            e_props = {'name': 'string',
                       'value': 'string'}
            eb.template(func='element_prop', props=e_props, **dict(name=k, value=v))

    def arguments(self, parent, props, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="ArgumentsPanel",
                               class_name='Arguments',
                               test_name="User Defined Variables",
                               prop_class_name='Arguments',
                               enabled='true')
        tree = fb.template(func='arguments', props={}, **kwargs)
        cp = self.elm.collection_prop(parent=tree, attributes=Attributes(name='Arguments.arguments'))
        for k, v in props.items():
            eb = JMeterTreeBuilder(parent=cp,
                                   name=k,
                                   element_type='Argument',
                                   prop_class_name='Argument')
            e_props = {'name': 'string',
                       'value': 'string',
                       'metadata': 'string'}
            eb.template(func='element_prop', props=e_props, **dict(name=k, value=v, metadata='='))

    def result_collector(self, parent, config, **kwargs):
        fb = JMeterTreeBuilder(parent=parent,
                               gui_class="ViewResultsFullVisualizer",
                               class_name='ResultCollector',
                               test_name="View Results Tree",
                               prop_class_name='ResultCollector',
                               enabled='true')
        props = {'error_logging': 'bool'}
        tree = fb.template(func='result_collector', props=props, **kwargs)
        self.elm.string_prop(parent=tree, attributes=Attributes(name="filename"))
        obj = self.elm.obj_prop(parent=tree)
        self.elm.custom_element(parent=obj, tag='name', text='saveConfig')
        obj_value = self.elm.custom_element(parent=obj, tag='value',
                                            attributes=Attributes(**{"class": "SampleSaveConfiguration"}))
        for k, v in config.items():
            self.elm.custom_element(parent=obj_value, tag=k, text=v)
