#include "strela.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include "../../../../runtime/workspace_pool.h"

#include <dlpack/dlpack.h>

#include <cstdlib>
#include <cstring>

#define PRINT do { LOG(INFO) << __func__; } while (0)

// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c#comment73511086_2745086
static size_t
ceil_div(size_t x, size_t y) {
  return x == 0 ? 0 : 1 + ((x - 1) / y);
}

namespace tvm {
namespace runtime {

class MyDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {
    PRINT;
    // cudaSetDevice
    this->dev = strela_dev_init(dev.device_id);
    if (!strela_dev_ok(this->dev)) {
      LOG(FATAL) << "Unable to set device " << dev.device_id << " because "
      << strela_dev_get_err(this->dev).errnum;
    }
  }

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    PRINT;
    int value = 0;
    switch (kind) {
      case kExist: {
        // cudaGetDeviceCount
        unsigned count = 0;
        int err = strela_device_count(&count);
        if (err == -1) {
          LOG(WARNING) << "The STRELA device count is incomplete!";
        }
        value = err != -1 && dev.device_id < static_cast<int>(count);
      }
    }
    *rv = value;
  }

  void *AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final {
    PRINT;
    // cudaMalloc
#if 0
    if (type_hint != DLDataType{kDLInt, 32, 1}) {
      LOG(FATAL) << "STRELA can only allocate int32 but a " << type_hint
        << " was given.";
    }
#endif
    if (alignment % sizeof (strela_word) != 0) {
      LOG(FATAL) << "STRELA can only allocate 4 bytes aligned data but "
        << alignment << " was requested.";
    }
    this->dev = strela_dev_init(dev.device_id);
    size_t size_word = ceil_div(size, sizeof (strela_word));
    strela_buffer buf = strela_buffer_alloc(this->dev, size_word);
    if (!buf.valid) {
      LOG(FATAL) << "Unable to allocate " << size << " bytes on STRELA";
    }
    return strela_buffer_to_ptr(this->dev, buf);
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    PRINT;
    strela_buffer_free(this->dev, strela_buffer_from_ptr(this->dev, ptr));
  }

#if 0
  TVMStreamHandle CreateStream(Device dev) {
    PRINT;
    return nullptr;
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    PRINT;
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    PRINT;
  }
#endif

  void StreamSync(Device dev, TVMStreamHandle stream) final { PRINT; }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  void CopyDataFromTo(
    const void* from, size_t from_offset,
    void* to, size_t to_offset, size_t size,
    Device dev_from, Device dev_to,
    DLDataType type_hint, TVMStreamHandle stream
  ) final {
    PRINT;
    std::memcpy(
      static_cast<unsigned char*>(to) + to_offset,
      static_cast<const unsigned char*>(from) + from_offset,
      size
    );
  }

#if 0
  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }
#endif

  strela_dev *dev;

  static MyDeviceAPI *Global() {
    PRINT;
    static auto *inst = new MyDeviceAPI();

    ffi::Any rv;
    inst->GetAttr(Device{kDLExtDev, 0}, kExist, &rv);

    int exists = rv.cast<int>();
    if (exists) {
      inst->dev = strela_dev_init(0);
    } else {
      LOG(FATAL) << "Cannot initialize MyDeviceAPI: STRELA device 0 does not exist.";
    }

    return inst;
  }

};

class MyThreadEntry {
 public:
  // The TVM runtime should use the WorkspacePool to allocate auxiliary data
  // needed by the given layer (everything that is not input and output buffers)
  // e.g. when performing im2col the "unrolled" input image. Give the algorithms
  // implementable in STRELA this should not be needed.
  WorkspacePool pool;

  MyThreadEntry() : pool(kDLExtDev, MyDeviceAPI::Global()) {}

  static MyThreadEntry* ThreadLocal() {
    static thread_local MyThreadEntry inst;
    return &inst;
  }
};

void* MyDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  PRINT;
  return MyThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
}

void MyDeviceAPI::FreeWorkspace(Device dev, void* data) {
  PRINT;
  MyThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def_packed("device_api.ext_dev", [](ffi::PackedArgs args, ffi::Any* rv) {
      PRINT;
      DeviceAPI* ptr = MyDeviceAPI::Global();
      *rv = static_cast<void*>(ptr);
    })
    .def("runtime.zero_copy_cpu_view", [](Tensor ext_array) -> Tensor {
      // NOTE: LLM generated, not user if this works correctly.

      // 1. Copy the DLTensor struct from the accelerator array
      DLTensor cpu_tensor = *ext_array.operator->();

      // 2. Spoof the device type so LLVM TIR kernels allow it
      cpu_tensor.device = Device{kDLCPU, 0};

      // 3. Wrap it in a DLManagedTensor to handle lifecycle safely
      DLManagedTensor* managed = new DLManagedTensor();
      managed->dl_tensor = cpu_tensor;
      managed->manager_ctx = new Tensor(ext_array); // Keep original array alive

      managed->deleter = [](DLManagedTensor* self) {
        // Drop the ref-count to the original array when the view dies
        delete static_cast<Tensor*>(self->manager_ctx);
        delete self;
      };

      return Tensor::FromDLPack(managed);
    });
}

}  // namespace runtime
}  // namespace tvm

extern "C" {
TVM_DLL void my_func(double *x, double *y, size_t len);

void my_func(double *x, double *y, size_t len) {
  PRINT;
  for (size_t i = 0; i < len; i++) {
    y[i] = x[i] + 1;
  }
}
}
