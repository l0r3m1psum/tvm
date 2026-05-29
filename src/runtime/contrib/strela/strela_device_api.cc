#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/function.h>
#include "../../workspace_pool.h"

#include <dlpack/dlpack.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#define PRINT do { std::cout << __func__ << '\n'; } while (0)

#if defined(_WIN32) || defined(__CYGWIN__)
  #include <malloc.h>
#else
  #include <stdlib.h>

  static void *
  _aligned_malloc(size_t size, size_t alignment) {
    void *ptr = NULL;

    // posix_memalign requires the alignment to be a power of 2 and a multiple
    // of sizeof (void*).
    size_t new_alignment = sizeof (void*);
    while (new_alignment < alignment) {
      new_alignment <<= 1;
    }

    if (posix_memalign(&ptr, new_alignment, size) != 0) {
      ptr = NULL; // This should not be necessary but we do it just in case.
    }
    return ptr;
  }

  static void
  _aligned_free(void* ptr) {
    free(ptr);
  }
#endif

namespace tvm {
namespace runtime {

class MyDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { PRINT; }

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    PRINT;
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void *AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final {
    PRINT;
    return _aligned_malloc(size, alignment);
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    PRINT;
    _aligned_free(ptr);
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

  static MyDeviceAPI *Global() {
    PRINT;
    static auto *inst = new MyDeviceAPI();
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
  .def_packed(
    "device_api.ext_dev",
    [](ffi::PackedArgs args, ffi::Any* rv) {
      DeviceAPI* ptr = MyDeviceAPI::Global();
      *rv = static_cast<void*>(ptr);
    }
  )
  .def(
    "ext_dev.zero_copy_cpu_view",
    [](Tensor ext_array) -> Tensor {
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
    }
  );
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
