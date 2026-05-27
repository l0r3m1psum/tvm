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

          int64_t num_elements = 1;
          for (int dim = 0; dim < input->ndim; ++dim) {
            num_elements *= input->shape[dim];
          }

          const float* input_data = static_cast<const float *>(input->data);
          float* out_data = static_cast<float *>(output->data);

          for (int64_t i = 0; i < num_elements; ++i) {
            out_data[i] = 23.f; // std::max(0.0f, input_data[i]);
          }
        }
      }
    }

    LOG(INFO) << "NPU execution completed";
  }

 private:

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
