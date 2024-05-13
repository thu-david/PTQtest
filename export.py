import onnx
import pycuda.driver as cuda
import tensorrt as trt
import h5py
import numpy as np
from typing import Dict, Optional, Sequence, Union
import os
import re,sys


DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

class HDF5Calibrator(trt.IInt8Calibrator):
    """HDF5 calibrator.

    Args:
        calib_file (str | h5py.File):  Input calibration file.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        model_type (str): Input model type, defaults to 'end2end'.
        device_id (int): Cuda device id, defaults to 0.
        algorithm (trt.CalibrationAlgoType): Calibration algo type, defaults
            to `trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2`.
    """

    def __init__(
            self,
            calib_file: Union[str, h5py.File],
            input_shapes: Dict[str, Sequence[int]],
            model_type: str = 'end2end',
            device_id: int = 0,
            algorithm: trt.CalibrationAlgoType = DEFAULT_CALIBRATION_ALGORITHM,
            **kwargs):
        super().__init__()

        if isinstance(calib_file, str):
            calib_file = h5py.File(calib_file, mode='r')

        assert 'calib_data' in calib_file
        calib_data = calib_file['calib_data']
        assert model_type in calib_data
        calib_data = calib_data[model_type]

        self.calib_file = calib_file
        self.calib_data = calib_data
        self.device_id = device_id
        self.algorithm = algorithm
        self.input_shapes = input_shapes
        self.kwargs = kwargs

        # create buffers that will hold data batches
        self.buffers = dict()

        self.count = 0
        first_input_group = calib_data[list(calib_data.keys())[0]]
        self.dataset_length = len(first_input_group)
        self.batch_size = first_input_group['0'].shape[0]

    def __del__(self):
        """Close h5py file if necessary."""
        if hasattr(self, 'calib_file'):
            self.calib_file.close()

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:

            ret = []
            for name in names:
                input_group = self.calib_data[name]
                data_np = input_group[str(self.count)][...].astype(np.float32)

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None

    def get_algorithm(self) -> trt.CalibrationAlgoType:
        """Get Calibration algo type.

        Returns:
            trt.CalibrationAlgoType: Calibration algo type.
        """
        return self.algorithm

    def get_batch_size(self) -> int:
        """Get batch size.

        Returns:
            int: An integer represents batch size.
        """
        return self.batch_size

    def read_calibration_cache(self, *args, **kwargs):
        """Read calibration cache.

        Notes:
            No need to implement this function.
        """
        pass

    def write_calibration_cache(self, cache, *args, **kwargs):
        """Write calibration cache.

        Notes:
            No need to implement this function.
        """
        pass



class HDF5CalibratorBEVDet(HDF5Calibrator):

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:
            if self.count % 100 == 0:
                print('%d/%d' % (self.count, self.dataset_length))
            ret = []
            for name in names:
                input_group = self.calib_data[name]
                if name == 'img':
                    data_np = input_group[str(self.count)][...].astype(
                        np.float32)
                else:
                    data_np = input_group[str(self.count)][...].astype(
                        np.int32)

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None


def save(engine: trt.ICudaEngine, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to be serialized.
        path (str): The absolute disk path to write the engine.
    """
    with open(path, "wb") as f:
        # # Serialize the engine to memory
        # serialized_engine = engine.serialize()
        # # Write the serialized engine to a file
        f.write(engine)  # 注意这里使用.data()来访问字节数组


def search_cuda_version() -> str:
    """try cmd to get cuda version, then try `torch.cuda`

    Returns:
        str: cuda version, for example 10.2
    """

    version = None

    pattern = re.compile(r'[0-9]+\.[0-9]+')
    platform = sys.platform.lower()

    def cmd_result(txt: str):
        cmd = os.popen(txt)
        return cmd.read().rstrip().lstrip()

    if platform == 'linux' or platform == 'darwin' or platform == 'freebsd':  # noqa E501
        version = cmd_result(
            " nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' "  # noqa E501
        )
        if version is None or pattern.match(version) is None:
            version = cmd_result(
                " nvidia-smi  | grep CUDA | awk '{print $9}' ")

    elif platform == 'win32' or platform == 'cygwin':
        # nvcc_release = "Cuda compilation tools, release 10.2, V10.2.89"
        nvcc_release = cmd_result(' nvcc --version | find "release" ')
        if nvcc_release is not None:
            result = pattern.findall(nvcc_release)
            if len(result) > 0:
                version = result[0]

        if version is None or pattern.match(version) is None:
            # nvidia_smi = "| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |" # noqa E501
            nvidia_smi = cmd_result(' nvidia-smi | find "CUDA Version" ')
            result = pattern.findall(nvidia_smi)
            if len(result) > 2:
                version = result[2]

    if version is None or pattern.match(version) is None:
        try:
            import torch
            version = torch.version.cuda
        except Exception:
            pass

    return version


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    import os
    old_cuda_device = os.environ.get('CUDA_DEVICE', None)
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    # load_tensorrt_plugin()
    # import ctypes
    # ctypes.CDLL("/workspace/TRTPlugin/lib/libdeploy_tensorrt_ops.so")
    # plugins = [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]
    # create builder and network
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)


    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # # config builder
    # if version.parse(trt.__version__) < version.parse('8'):
    #     builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        # if version.parse(trt.__version__) < version.parse('8'):
        #     builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5CalibratorBEVDet(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        # if version.parse(trt.__version__) < version.parse('8'):
        #     builder.int8_mode = int8_mode
        #     builder.int8_calibrator = config.int8_calibrator

    # create engine
    engine = builder.build_serialized_network(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, 'my.engine')
    return engine


def main():
    import ctypes
    ctypes.CDLL("/workspace/TRTPlugin/lib/libdeploy_tensorrt_ops.so")
    plugins = [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]

    input_shapes = {
        'img': {'min_shape': [4, 3, 352, 640], 'opt_shape': [4, 3, 352, 640], 'max_shape': [4, 3, 352, 640]},
        'ranks_depth': {'min_shape': [26217], 'opt_shape': [26217], 'max_shape': [26217]},
        'ranks_feat': {'min_shape': [26217], 'opt_shape': [26217], 'max_shape': [26217]},
        'ranks_bev': {'min_shape': [26217], 'opt_shape': [26217], 'max_shape': [26217]},
        'interval_starts': {'min_shape': [3586], 'opt_shape': [3586], 'max_shape': [3586]},
        'interval_lengths': {'min_shape': [3586], 'opt_shape': [3586], 'max_shape': [3586]}
    }


    from_onnx(
        './bev.onnx',
        fp16_mode=True,
        int8_mode=True,
        int8_param=dict(
            calib_file=os.path.join('/workspace/workdir', 'calib.h5'),
            model_type='end2end'),
        max_workspace_size=1 << 30,
        input_shapes=input_shapes)


if __name__ == '__main__':
    print(trt.__version__)
    main()