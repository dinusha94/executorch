

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

// #include "arm_perf_monitor.h"

#include "model_pte.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferCleanup;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;


#define METHOD_ALLOCATOR_POOL_SIZE (70 * 1024 * 1024)
unsigned char __attribute__((
    section("network_model_sec"),
    aligned(16))) method_allocation_pool[METHOD_ALLOCATOR_POOL_SIZE];

const size_t temp_allocation_pool_size = 1 * 1024 * 1024;
unsigned char __attribute__((
    section("network_model_sec"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
   __builtin_trap();

}


namespace {

Result<BufferCleanup> prepare_input_tensors(
    Method& method,
    MemoryAllocator& allocator,
    std::vector<std::pair<char*, size_t>>& input_buffers) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;

  void** inputs =
      static_cast<void**>(allocator.allocate(num_inputs * sizeof(void*)));

  ET_CHECK_OR_RETURN_ERROR(
      inputs != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for pointers to input buffers.");

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    // Input is a tensor. Allocate a buffer for it.
    void* data_ptr = allocator.allocate(tensor_meta->nbytes());
    ET_CHECK_OR_RETURN_ERROR(
        data_ptr != nullptr,
        MemoryAllocationFailed,
        "Could not allocate memory for input buffers.");
    inputs[num_allocated++] = data_ptr;

    Error err = Error::Ok;
    if (input_buffers.size() > 0) {
      auto [buffer, buffer_size] = input_buffers.at(i);
      if (buffer_size != tensor_meta->nbytes()) {
        ET_LOG(
            Error,
            "input size (%d) and tensor size (%d) missmatch!",
            buffer_size,
            tensor_meta->nbytes());
        err = Error::InvalidArgument;
      } else {
        ET_LOG(Info, "Copying read input to tensor.");
        std::memcpy(data_ptr, buffer, buffer_size);
      }
    }

    TensorImpl impl = TensorImpl(
        tensor_meta.get().scalar_type(),
        tensor_meta.get().sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta.get().sizes().data()),
        data_ptr,
        const_cast<TensorImpl::DimOrderType*>(
            tensor_meta.get().dim_order().data()));
    Tensor t(&impl);

    // If input_buffers.size <= 0, we don't have any input, fill t with 1's.
    if (input_buffers.size() <= 0) {
      for (size_t j = 0; j < t.numel(); j++) {
        switch (t.scalar_type()) {
          case ScalarType::Int:
            t.mutable_data_ptr<int>()[j] = 1;
            break;
          case ScalarType::Float:
            t.mutable_data_ptr<float>()[j] = 1.;
            break;
        }
      }
    }

    err = method.set_input(t, i);

    if (err != Error::Ok) {
      ET_LOG(
          Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
      // The BufferCleanup will free the inputs when it goes out of scope.
      BufferCleanup cleanup({inputs, num_allocated});
      return err;
    }
  }
  return BufferCleanup({inputs, num_allocated});
}

} // namespace



int main() {

    executorch::runtime::runtime_init();

    std::vector<std::pair<char*, size_t>> input_buffers;
    size_t pte_size = sizeof(model_pte);

    ET_LOG(Info, "Model in %p %c", model_pte, model_pte[0]);
    auto loader = BufferDataLoader(model_pte, pte_size);
    ET_LOG(Info, "Model PTE file loaded. Size: %lu bytes.", pte_size);
    Result<Program> program = Program::load(&loader);
    if (!program.ok()) {
        ET_LOG(
            Info,
            "Program loading failed @ 0x%p: 0x%" PRIx32,
            model_pte,
            program.error());
    }

    ET_LOG(Info, "Model buffer loaded, has %lu methods", program->num_methods());



     const char* method_name = nullptr;
    {
        const auto method_name_result = program->get_method_name(0);
        ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
        method_name = *method_name_result;
    }
    
    ET_LOG(Info, "Running method %s", method_name);

    Result<MethodMeta> method_meta = program->method_meta(method_name);
    if (!method_meta.ok()) {
        ET_LOG(
            Info,
            "Failed to get method_meta for %s: 0x%x",
            method_name,
            (unsigned int)method_meta.error());
    }

    MemoryAllocator method_allocator(
        METHOD_ALLOCATOR_POOL_SIZE, method_allocation_pool);

    std::vector<uint8_t*> planned_buffers; // Owns the memory
    std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
    size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
        size_t buffer_size =
            static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
        ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);

        /* Move to it's own allocator when MemoryPlanner is in place. */
        uint8_t* buffer =
            reinterpret_cast<uint8_t*>(method_allocator.allocate(buffer_size));
        planned_buffers.push_back(buffer);
        planned_spans.push_back({planned_buffers.back(), buffer_size});
    }

    HierarchicalAllocator planned_memory(
        {planned_spans.data(), planned_spans.size()});

    MemoryAllocator temp_allocator(
        temp_allocation_pool_size, temp_allocation_pool);

    MemoryManager memory_manager(
        &method_allocator, &planned_memory, &temp_allocator);

    Result<Method> method = program->load_method(method_name, &memory_manager);
    if (!method.ok()) {
        ET_LOG(
            Info,
            "Loading of method %s failed with status 0x%" PRIx32,
            method_name,
            method.error());
    }
    ET_LOG(Info, "Method loaded.");

    ET_LOG(Info, "Preparing inputs...");

    auto inputs =
        ::prepare_input_tensors(*method, method_allocator, input_buffers);

    if (!inputs.ok()) {
        ET_LOG(
            Info,
            "Preparing inputs tensors for method %s failed with status 0x%" PRIx32,
            method_name,
            inputs.error());
    }
    ET_LOG(Info, "Input prepared.");

    ET_LOG(Info, "Starting the model execution...");
  
    Error status = method->execute();

    if (status != Error::Ok) {
        ET_LOG(
            Info,
            "Execution of method %s failed with status 0x%" PRIx32,
            method_name,
            status);
    } else {
        ET_LOG(Info, "Model executed successfully.");
    }

    std::vector<EValue> outputs(method->outputs_size());
    ET_LOG(Info, "%zu outputs: ", outputs.size());
    status = method->get_outputs(outputs.data(), outputs.size());
    ET_CHECK(status == Error::Ok);

    for (int i = 0; i < outputs.size(); ++i) {
        Tensor t = outputs[i].toTensor();
    
        // The output might be collected and parsed so printf() is used instead
        // of ET_LOG() here
        for (int j = 0; j < outputs[i].toTensor().numel(); ++j) {
            if (t.scalar_type() == ScalarType::Int) {
                printf(
                    "Output[%d][%d]: %d\n",
                    i,
                    j,
                    outputs[i].toTensor().const_data_ptr<int>()[j]);
            } else {
                printf(
                    "Output[%d][%d]: %f\n",
                    i,
                    j,
                    outputs[i].toTensor().const_data_ptr<float>()[j]);
            }
        }
    
    }
    out:
        ET_LOG(Info, "Program complete, exiting.");
    
        ET_LOG(Info, "\04");

        ET_LOG(Info, "Hellow me!");

        return 0;
}
