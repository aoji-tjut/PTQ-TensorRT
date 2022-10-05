#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import time
from cuda import cudart
import cv2
import numpy as np
import os
import tensorrt as trt
import torch
from torchvision import models, transforms, datasets
from tqdm import tqdm
import calibrator

torch.backends.cudnn.deterministic = True
batch_size = 1
channel = 3
nHeight = 32
nWidth = 32
onnxFile = "./model.onnx"
trtFile = "./model.plan"
inferenceImage = "/data/data/cifar100/cali/1.jpg"

isFP16Mode = False
isINT8Mode = True
nCalibration = 3
cacheFile = "./int8.cache"
calibrationDataPath = "/data/data/cifar100/cali/"

os.system("rm -rf ./*.onnx ./*.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

torch_model = models.resnet18(num_classes=100)
ckpt = torch.load("/data/data/cifar100/resnet18.pth", map_location="cpu")
msg = torch_model.load_state_dict(ckpt, strict=False)
print(msg)

# 导出模型为 onnx文件
torch_model.eval()
torch.onnx.export(torch_model,
                  torch.randn(1, channel, nHeight, nWidth, requires_grad=True),
                  onnxFile,
                  input_names=["x"],
                  output_names=["y"],
                  dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
                  verbose=False)
print("Succeeded converting model into onnx!")

# TensorRT 中加载onnx 创建engine
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
config.max_workspace_size = 1 << 30
if isFP16Mode:
    config.flags = 1 << int(trt.BuilderFlag.FP16)
if isINT8Mode:
    config.flags = 1 << int(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration,
                                                     (1, channel, nHeight, nWidth), cacheFile)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, (1, channel, nHeight, nWidth), (batch_size, channel, nHeight, nWidth),
                  (256, channel, nHeight, nWidth))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_binding_shape(0, [1, channel, nHeight, nWidth])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(engine.num_bindings):
    print("Bind[%d]:i[%d]->" % (i, i) if engine.binding_is_input(i) else "Bind[%d]:o[%d]->" % (i, i - nInput),
          engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i),
          engine.get_binding_name(i))

data = cv2.imread(inferenceImage, cv2.COLOR_BGR2RGB).astype(np.float32).reshape(1, channel, nHeight, nWidth)
bufferH = []
bufferH.append(data)
for i in range(nOutput):
    bufferH.append(
        np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))

bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nOutput):
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

print("(x) inputH0:", bufferH[0].shape)
print("(y) outputH0:", bufferH[-1].shape)

for buffer in bufferD:
    cudart.cudaFree(buffer)

print("Succeeded running model in TensorRT!")

# Test--------------------------------------------------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_set = datasets.CIFAR100(root='/data/data/cifar100', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=32)

device = "cuda"
torch_model.eval()
torch_model.to(device)
torch_model.eval()

all_time = 0.0
running_corrects = 0
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    t1 = time.time()
    outputs = torch_model(inputs)
    t2 = time.time()
    all_time += (t2 - t1)

    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
test_accuracy = running_corrects / len(test_loader.dataset)
print("test_accuracy:", test_accuracy)
print("torch time:", all_time)

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_binding_shape(0, [batch_size, channel, nHeight, nWidth])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

all_time = 0.0
running_corrects = 0
for inputs, labels in tqdm(test_loader):
    inputs = inputs.numpy()
    labels = labels.numpy()

    # set input and output
    bufferH = []
    bufferH.append(inputs)
    for i in range(nOutput):
        bufferH.append(
            np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))

    # cuda malloc
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # cpu to cuda
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # forward
    t1 = time.time()
    context.execute_v2(bufferD)
    t2 = time.time()
    all_time += (t2 - t1)

    # cuda to cpu
    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    y = bufferH[-1]

    # softmax
    y = np.exp(y)
    softmax = y / np.sum(y, axis=1).reshape(batch_size, 1)

    preds = np.argmax(softmax, axis=1)
    running_corrects += np.sum(preds == labels)

    for buffer in bufferD:
        cudart.cudaFree(buffer)

test_accuracy = running_corrects / len(test_loader.dataset)
print("test_accuracy:", test_accuracy)
print("trt time:", all_time)
