import time
import joblib
import pandas as pd
import numpy as np

from core.synthetic_gen.base import SyntheticDataGen
from core.load_test.load_test import ModelLoadTest
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp


def synth_data_gen():
    synth = SyntheticDataGen()

    # data = pd.read_csv(r'/Users/Vishvig/Desktop/Education/'
    #                    r'UoB/Courses/Dissertation/LV=/PoC2Prod/datasets/continuous_w_discreteTarget/WineQT.csv')
    # data = pd.read_csv(r'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/PoC2Prod/'
    #                    r'datasets/imbalanced_dataset/aug_train.csv')[0:10000]
    # data = pd.read_csv(r'/Users/Vishvig/Desktop/Education/'
    #                    r'UoB/Courses/Dissertation/LV=/PoC2Prod/datasets/dataset_w_na/dataset.csv')
    # data = pd.read_csv(r'/Users/Vishvig/Desktop/Education/'
    #                    r'UoB/Courses/Dissertation/LV=/PoC2Prod/datasets/timeseries_dataset/timeseries_dataset.csv')[0:5000]
    data = pd.read_csv(r'/Users/Vishvig/Desktop/Education/'
                       r'UoB/Courses/Dissertation/LV=/PoC2Prod/datasets/timeseries_w_discrete_continuous/Genesis_AnomalyLabels.csv')[
           0:5000]
    print(data.shape)
    # synth.fit(data=data,
    #           discrete=['Gender', 'Age', 'Driving_License',
    #                     'Region_Code', 'Previously_Insured', 'Vehicle_Age',
    #                     'Vehicle_Damage', 'Policy_Sales_Channel', 'Vintage', 'Response'],
    #           ids=['id'])
    # synth.fit(data=data,
    #           discrete=['quality'],
    #           ids=['Id'])
    # synth.fit(data=data,
    #           discrete=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    #                     'work_type', 'Residence_type', 'smoking_status', 'stroke'],
    #           ids=['id'])
    # synth.fit(data=data)
    synth.fit(data=data,
              discrete=['Label', 'MotorData.Motor_Pos1reached', 'MotorData.Motor_Pos2reached',
                        'MotorData.Motor_Pos3reached', 'Motor_Pos4reached',
                        'NVL_Recv_Ind.GL_Metall', 'NVL_Recv_Ind.GL_NonMetall', 'NVL_Recv_Storage.GL_I_ProcessStarted',
                        'NVL_Recv_Storage.GL_I_Slider_IN', 'NVL_Recv_Storage.GL_I_Slider_OUT',
                        'NVL_Recv_Storage.GL_LightBarrier', 'NVL_Send_Storage.ActivateStorage',
                        'PLC_PRG.Gripper', 'PLC_PRG.MaterialIsMetal'])
    xn = synth.sample(n=5000)
    # xn.to_csv(r'/Users/Vishvig/Desktop/Education/'
    #           r'UoB/Courses/Dissertation/LV=/PoC2Prod/datasets/continuous_w_discreteTarget/synthetic.csv',
    #           index=False)
    # xn.to_csv(r'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/PoC2Prod/datasets'
    #           r'/imbalanced_dataset/synthetic.csv',
    #           index=False)
    # xn.to_csv(r'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/PoC2Prod/datasets'
    #           r'/dataset_w_na/synthetic.csv',
    #           index=False)
    # xn.to_csv(r'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/PoC2Prod/datasets'
    #           r'/timeseries_dataset/synthetic.csv',
    #           index=False)
    xn.to_csv(r'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/PoC2Prod/datasets'
              r'/timeseries_w_discrete_continuous/synthetic.csv',
              index=False)


def load_test(data, skip_runs=None):
    # Initialize the config variables
    # instance_counts = [1, 2, 3, 5]
    # ramp_up_time = [1, 5, 10, 30, 60]
    # num_threads = [100, 200, 500, 750, 1000]
    # resources = [{'cpu': '1:2', 'memory': '1Gi:2Gi'},
    #              {'cpu': '1:2', 'memory': '2Gi:4Gi'},
    #              {'cpu': '2:4', 'memory': '2Gi:4Gi'},
    #              {'cpu': '500m:1000m', 'memory': '1Gi:2Gi'}]

    instance_counts = [1, 2, 3, 4, 5]
    ramp_up_time = [1, 5, 10, 30, 60]
    num_threads = [100, 200, 500, 1000]
    loops = [1, 2, 5, 10]
    resources = [{'cpu': '1:2', 'memory': '2Gi:4Gi'},
                 {'cpu': '500m:1', 'memory': '1Gi:2Gi'},
                 {'cpu': '2:4', 'memory': '2Gi:4Gi'}]

    if skip_runs is None:
        skip_runs = 0

    # initialize the load test cluster
    lt = ModelLoadTest(model_info={"subscription_id": "3ea884f0-5248-4c66-99b1-2c7f0a7df066",
                                   "resource_grp": "vravishankar",
                                   "ml_workspace": "modelPerformanceTesting",
                                   "endpoint_name": "k8s-endpoint-logistic-regression",
                                   "compute_name": "sandbox1",
                                   'deployment_name': 'third'},
                       run_id=skip_runs)
    lt.start(engine_instances=0,
             az_ml_k8s_config={"cluster_name": "modelperformancetestingsandbox"},
             load_test_k8s_config={"namespace": "sandbox",
                                   "cluster_name": "jmeterCluster",
                                   "resources": {"cpu": {"requests": '2',
                                                         "limit": '4'},
                                                 "memory": {"requests": '8Gi',
                                                            "limit": '16Gi'}
                                                 }
                                   })

    # Run load test for different configurations
    count = 0
    for i in instance_counts:
        for t in num_threads:
            for r in ramp_up_time:
                for l in loops:
                    for res in resources:
                        if count < skip_runs:
                            count += 1
                            continue
                        results_path = f'/Users/Vishvig/Desktop/Education/UoB/Courses/Dissertation/LV=/' \
                                       f'PoC2Prod/codebase/results/'
                        cpu = res.get('cpu', None)
                        memory = res.get('memory', None)
                        gpu = res.get('gpu', None)
                        cpu_requests, cpu_limits, memory_requests, memory_limits, gpu_requests, gpu_limits = \
                            None, None, None, None, None, None
                        if cpu is not None:
                            try:
                                cpu_requests = cpu.split(':')[0]
                                cpu_limits = cpu.split(':')[1]
                            except IndexError:
                                pass
                        if memory is not None:
                            try:
                                memory_requests = memory.split(':')[0]
                                memory_limits = memory.split(':')[1]
                            except IndexError:
                                pass
                        if gpu is not None:
                            try:
                                gpu_requests = gpu.split(':')[0]
                                gpu_limits = gpu.split(':')[1]
                            except IndexError:
                                pass
                        lt.run_test(
                            results_path=results_path,
                            data=data,
                            threads_info={'num_threads': int(round(t / l)),
                                          'ramp_time': r,
                                          'loops': l},
                            resources=dict(requests=dict(cpu=cpu_requests, memory=memory_requests, gpu=gpu_requests),
                                           limits=dict(cpu=cpu_limits, memory=memory_limits, gpu=gpu_limits)),
                            instance_count=i)
                        print('--------------------------------------------')
    lt.end()


def to_numpy(df, preds, trgts, encs=None):
    columns = list(df.columns)
    X = np.empty(df[[i for k, v in preds.items() for i in v]].shape)
    Y = np.empty(df[[i for k, v in trgts.items() for i in v]].shape)

    for i, col in enumerate(preds['continuous']):
        X[:, columns.index(col)] = df[col]
    for i, col in enumerate(trgts['continuous']):
        Y[:, i] = df[col]

    if encs is None:
        encs = dict()
    for i, col in enumerate(preds['categorical']):
        if col not in encs:
            enc = LabelEncoder()
            enc.fit(df[col])
            encs[col] = enc
        else:
            enc = encs[col]
        X[:, columns.index(col)] = enc.transform(df[col])
    for i, col in enumerate(trgts['categorical']):
        if col not in encs:
            enc = LabelEncoder()
            enc.fit(df[col])
            encs[col] = enc
        else:
            enc = encs[col]
        Y[:, i] = enc.transform(df[col])

    return X, Y, encs


if __name__ == '__main__':
    # Generating synthetic data
    synth_data_gen()

    # Pre-processing synthetic data for deployed models
    # predictors = {'categorical': ['Gender', 'Age', 'Driving_License',
    #                               'Region_Code', 'Previously_Insured', 'Vehicle_Age',
    #                               'Vehicle_Damage', 'Policy_Sales_Channel', 'Vintage'],
    #               'continuous': ['Annual_Premium']}
    # targets = {'categorical': ['Response'],
    #            'continuous': []}
    # pre_process_model = joblib.load(r"../model/logistic_regression_pre_process.pkl")
    # test_df = pd.read_csv('../datasets/imbalanced_dataset/synthetic1.csv', index_col=False)
    # test_df = test_df[pre_process_model['columns']]
    # test_X, test_Y, test_encs = to_numpy(test_df,
    #                                      preds=predictors,
    #                                      trgts=targets,
    #                                      encs=pre_process_model["encs"])
    #
    # # Passing synthetic data to load testing
    # test_X = test_X[0:10000]
    # load_test(data=test_X.tolist(), skip_runs=513)
