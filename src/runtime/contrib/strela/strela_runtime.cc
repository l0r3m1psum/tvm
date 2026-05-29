/*!
 * \file src/runtime/contrib/strela/strela_runtime.cc
 * \brief STRELA runtime
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "strela.h"

static const uint32_t relu_kernel[STRELA_KERNEL_SIZE] = {
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00004083, 0x20CC0300, 0x000000A0, 0x00000000, 0x00000000, // 4
  0x00000241, 0x020C0300, 0x00000099, 0x00000000, 0x00000000, // 0

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 9
  0x00000011, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 5
  0x00400008, 0x00000200, 0x00000000, 0x00000000, 0x00000000, // 1

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 10
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 6
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 2

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 11
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 7
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 3
};

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class STRELARuntime : public JSONRuntimeBase {
 public:
  STRELARuntime(const std::string& symbol_name, const std::string& graph_json,
                const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~STRELARuntime() = default;

  const char* kind() const override { return "strela_json"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";

    SetupConstants(consts);

    LOG(INFO) << "Initialization";

    unsigned which = 0;
    dev = strela_dev_init(which);
    if (!strela_dev_ok(dev)) {
      LOG(FATAL) << "Unable to initialized STRELA device " << which
        << " because " << strela_dev_get_err(dev).errnum;
    }

    AnalyzeGraphForOptimization();
  }

  void Run() override {
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const JSONGraphNode& node = nodes_[i];

      if (node.GetOpType() == "kernel") {
        std::string op_name = node.GetOpName();

        if (op_name.find("relu") != std::string::npos) {
          const std::vector<JSONGraphNodeEntry>& inputs = node.GetInputs();
          if (inputs.size() != 1) {
            LOG(FATAL) << "ReLU requires only 1 inputs.";
          }

          uint32_t input_id = inputs[0].id_;
          uint32_t output_id = i;

          const DLTensor *input = data_entry_[input_id];
          const DLTensor *output = data_entry_[output_id];

          DLDataType int32_dtype = {kDLInt, 32, 1};
          if (input->dtype != int32_dtype || output->dtype != int32_dtype) {
            LOG(FATAL) << "ReLU on STRELA works only in int32.";
          }

          int64_t num_elements = 1;
          for (int dim = 0; dim < input->ndim; ++dim) {
            num_elements *= input->shape[dim];
          }

          const int32_t *input_data = static_cast<const int32_t *>(input->data);
          int32_t *output_data = static_cast<int32_t *>(output->data);

          // TODO: assert that the tensors are contiguous (i.e. no strides)

          strela_kernel kernel = strela_kernel_alloc(dev);
          strela_kernel_set(dev, kernel, relu_kernel);
          strela_buffer input_buf = strela_buffer_alloc(dev, num_elements);
          strela_buffer output_buf = strela_buffer_alloc(dev, num_elements);
          strela_buffer_set(dev, input_buf, input_data);

          strela_conf conf = {
            .inp0_offset = input_buf.offset_words_from_base, .inp0_count = num_elements, .inp0_stride = sizeof (strela_word),
            .out0_offset = output_buf.offset_words_from_base, .out0_count = num_elements,
          };

          strela_config(dev, kernel, &conf);
          strela_execute(dev);
          strela_buffer_get(dev, output_buf, output_data);
          strela_buffer_free_all(dev);
          strela_kernel_free_all(dev);

          if (!strela_dev_ok(dev)) {
            LOG(FATAL) << "Unable to run ReLU on STRELA because " << strela_dev_get_err(dev).errnum;
          }

          // for (int64_t i = 0; i < num_elements; ++i) {
          //   output_data[i] = 23.f; // std::max(0.0f, input_data[i]);
          // }
        }
      }
    }

    LOG(INFO) << "NPU execution completed";
  }

 private:
  strela_dev *dev;

  void AnalyzeGraphForOptimization() {
    for (const JSONGraphNode& node : nodes_) {
      uint32_t num_output = node.GetNumOutput();
      std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
      std::string op_type = node.GetOpType();
      std::string op_name = node.GetOpName();
      ffi::Array<ffi::Array<int64_t>> op_shape = node.GetOpShape();
      ffi::Array<DLDataType> op_data_type = node.GetOpDataType();
      // from the public API there is no way to enumerate the Attrs
      if (node.HasAttr("T")) {
        auto dtype_iter = node.GetAttr<ffi::Array<ffi::String>>("T");
        if (!dtype_iter.empty()) {
          LOG(INFO) << "dtype: " << dtype_iter[0];
        }
      }

      LOG(INFO) << "num_output: " << num_output;
      // LOG(INFO) << "inputs: " << inputs;
      LOG(INFO) << "op_type: " << op_type;
      LOG(INFO) << "op_name: " << op_name;
      // LOG(INFO) << "op_shape: " << op_shape;
      // LOG(INFO) << "op_data_type: " << op_data_type;
    }
  }
};

ffi::Module STRELARuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<STRELARuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.STRELAJSONRuntimeCreate", STRELARuntimeCreate)
      .def("ffi.Module.load_from_bytes.strela_json",
           JSONRuntimeBase::LoadFromBytes<STRELARuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
