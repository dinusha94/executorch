import torch
from torchvision.models import mobilenet_v2  # @manual
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


# mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# mv2.eval()

# variant='edgeface_xs_gamma_06'
# model = torch.hub.load('otroshi/edgeface', variant, source='github', pretrained=True)
# model.eval()



# import torch
# from torch.export import export, ExportedProgram


# class SimpleConv(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv = torch.nn.Conv2d(
#             in_channels=3, out_channels=16, kernel_size=3, padding=1
#         )
#         self.relu = torch.nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         a = self.conv(x)
#         return self.relu(a)


# example_args = (torch.randn(1, 3, 256, 256),)
# aten_dialect: ExportedProgram = export(SimpleConv(), example_args)
# print(aten_dialect)


# # 2. to_edge: Make optimizations for Edge devices
# edge_program = to_edge(aten_dialect)

# # 3. to_executorch: Convert the graph to an ExecuTorch program
# executorch_program = edge_program.to_executorch()

# # 4. Save the compiled .pte program
# with open("./models/model.pte", "wb") as file:
#     file.write(executorch_program.buffer)

# import torch
# from torch.export import export
# from executorch.exir import to_edge

# # Start with a PyTorch model that adds two input tensors (matrices)
# class Add(torch.nn.Module):
#   def __init__(self):
#     super(Add, self).__init__()

#   def forward(self, x: torch.Tensor, y: torch.Tensor):
#       return x + y

# # 1. torch.export: Defines the program with the ATen operator set.
# aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))

# # 2. to_edge: Make optimizations for Edge devices
# edge_program = to_edge(aten_dialect)

# # 3. to_executorch: Convert the graph to an ExecuTorch program
# executorch_program = edge_program.to_executorch()

# # 4. Save the compiled .pte program
# with open("./models/add.pte", "wb") as file:
#     file.write(executorch_program.buffer)


import executorch.exir as exir

from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program

# Quantize model if required using the standard export quantizaion flow.
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
 
    quantizer = ArmQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    # make sure we can export to flat buffer
    return m

class CustomModel(nn.Module):
    def __init__(self, num_classes=299):
        super(CustomModel, self).__init__()
        
        # Load pre-trained MobileNetV2
        mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        mv2.eval()  # Set to evaluation mode to freeze batch norm layers

        # Remove the classifier (we'll replace it)
        self.base_model = mv2.features
        
        # Add custom layers on top
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dropout1 = nn.Dropout(0.3)
        self.embedding_layer = nn.Linear(1280, 512)  # 1280 is the final feature size of MobileNetV2
        self.dropout2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)  # Pass through MobileNetV2's feature extractor
        x = self.global_avg_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten the output to [batch_size, 1280]
        x = self.dropout1(x)
        x = self.embedding_layer(x)  # Embedding layer without activation
        x = self.dropout2(x)
        x = self.classifier(x)  # Final classification layer
        return x


class EmbeddingExtractor(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingExtractor, self).__init__()
        # Use only the layers up to the embedding layer
        self.features = nn.Sequential(
            base_model.base_model,  # MobileNetV2 feature extractor
            base_model.global_avg_pool,
            nn.Flatten(),  # Flatten directly within nn.Sequential
            base_model.embedding_layer  # Embedding layer without activation
        )
    
    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":

    torch.ops.load_library('/home/dinusha/executorch/cmake-out-aot-lib/kernels/quantized/libquantized_ops_aot_lib.so')

    # Load pre-trained weights from torchvision
    # pretrained_model = models.mobilenet_v2(pretrained=True)
    # pretrained_weights = pretrained_model.state_dict()

    # # Initialize your custom MobileNetV2 model
    # m = MobileNetV2(num_classes=1000)  # Ensure num_classes matches

    # Load the state dictionary into your model
    # m.load_state_dict(pretrained_weights)

    # m.eval()

    IMAGE_SIZE = 224  # Size of the input images
    NUM_CLASSES = 299

    # Initialize the model
    model = CustomModel(num_classes=NUM_CLASSES)

    # Wrap the model to extract embeddings
    embedding_model = EmbeddingExtractor(model)

    # Set to evaluation mode
    embedding_model.eval()

    # Example input
    sample_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)  # Batch size 1, 3 channels, IMAGE_SIZE x IMAGE_SIZE

    # Get embeddings
    embeddings = embedding_model(sample_input)


    example_inputs = (torch.randn((1, 3, 224, 224)),)


    model = torch._export.capture_pre_autograd_graph(embedding_model, example_inputs)

    # Quantize if required
    model = quantize(model, example_inputs)

    edge = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    edge = edge.to_backend(
            ArmPartitioner(
                ArmCompileSpecBuilder()
                .ethosu_compile_spec(
                    "ethos-u55-128",
                    system_config="Ethos_U55_High_End_Embedded",
                    memory_mode="Shared_Sram",
                    extra_flags="--debug-force-regor --output-format=raw",
                )
                .set_permute_memory_format(True)
                .set_quantize_io(True)
                .build()
            )
        )

    exec_prog = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )


    save_pte_program(exec_prog, "model-u55-128")