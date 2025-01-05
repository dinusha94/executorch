#include <iostream>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>


using namespace ::executorch::extension;
using namespace ::executorch::runtime;

using executorch::runtime::Result;


int main() {

    // Create a Module.
    Module module("../models/mobilenet_v2.pte");
    // Module module("../models/add.pte");

    // Wrap the input data with a Tensor.
    const int size = 1 * 3 * 224 * 224; // Total size of the array
    float input[size]; // Declare the array

    // Populate the array with a constant value of 2
    for (int i = 0; i < size; ++i) {
        input[i] = 2.0f;
    }

    auto tensor = from_blob(input, {1, 3, 224, 224});

    // float input1[1] = {2.0f};
    // float input2[1] = {2.0f};
    // auto tensor1 = from_blob(input1, {1});
    // auto tensor2 = from_blob(input2, {1});
    // std::vector<executorch::runtime::EValue> inputs = {tensor1, tensor2};
    // const auto error = module.set_inputs("forward", inputs);
    // if (error == Error::Ok){
    //     std::cout << "No error " << std::endl;
    // }

    // Perform an inference.
    const auto result = module.forward(tensor);
    // const auto result = module.forward();

    // Check for success or failure.
    if (result.ok()) {

        const auto& vec = *result;
        size_t length = vec.size();
        std::cout << "Vector length: " << length << std::endl;

        const auto num_dim = result->at(0).toTensor().dim();
        std::cout << "num dim : " << num_dim << std::endl; // 4

        const auto ze_dim_ = result->at(0).toTensor().size(0);
        std::cout << "size dim : " << ze_dim_ << std::endl;    // (1, 16, 256, 256)
        
        const auto num_elements = result->at(0).toTensor().numel();
        std::cout << "num elements in bits : " << num_elements << std::endl; // 256*256*16

        // Retrieve the output data.
        const auto output = result->at(0).toTensor().const_data_ptr<float>();

        std::cout << "Tensor data: ";
        for (int i = 0; i < num_elements; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;


        // int value = output[0]; 
        // std::cout << "output : " << value << std::endl;

    }

    return 0;
}
