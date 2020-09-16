import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from helpers.tokenization import BertTokenizer
from helpers.data_processing import read_squad_json,convert_example_to_features

# Todo: change base calibrator to IInt8EntropyCalibrator2
class BertCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self,squad_json,vocab_file,cache_file,batch_size,max_seq_length,num_inputs):
        super().__init__()
        self.cache_file=cache_file

        self.data=read_squad_json(squad_json)
        self.max_seq_length=max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        self.num_inputs = num_inputs
        self.tokenizer=BertTokenizer(vocab_file)
        self.doc_stride=128
        self.max_query_length=64

        self.device_inputs=[cuda.mem_alloc(self.max_seq_length*tft.int32.itemsize*self.batch_size) for binding in range(3)]

    def free(self):
        for dinput in self.device_inputs:
            dinput.free()

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self,names):
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        input_ids = []
        segment_ids = []
        input_mask = []
        for i in range(self.batch_size):
            example = self.data[self.current_index + i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            if len(input_ids) and len(segment_ids) and len(input_mask):
                input_ids = np.concatenate((input_ids, features[0].input_ids))
                segment_ids = np.concatenate((segment_ids, features[0].segment_ids))
                input_mask = np.concatenate((input_mask, features[0].input_mask))
            else:
                input_ids = features[0].input_ids
                segment_ids = features[0].segment_ids
                input_mask = features[0].input_mask

        cuda.memcpy_htod(self.device_inputs[0], input_ids.ravel())
        cuda.memcpy_htod(self.device_inputs[1], segment_ids.ravel())
        cuda.memcpy_htod(self.device_inputs[2], input_mask.ravel())

        self.current_index += self.batch_size
        return self.device_inputs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self,cache):
        with open(self.cache_file,'wb') as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0