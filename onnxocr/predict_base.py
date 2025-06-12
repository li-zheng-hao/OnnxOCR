import onnxruntime

class PredictBase(object):
    def __init__(self):
        pass

    def get_onnx_session(self, model_dir, use_gpu):
        # 使用gpu
        if use_gpu:
            # 修改cuDNN配置以避免Fallback模式
            providers =[
                ('CUDAExecutionProvider', {
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": True,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers =['CPUExecutionProvider']

        onnx_session = onnxruntime.InferenceSession(model_dir, None,providers=providers)

        # print("providers:", onnxruntime.get_device())
        return onnx_session


    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
