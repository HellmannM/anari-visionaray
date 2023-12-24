
#pragma once

// std
#include <vector>
// ours
#include "DeviceCopyableObjects.h"

#ifdef WITH_CUDA
// cuda
#include <cuda_runtime.h>
// visionaray
#include "visionaray/cuda/safe_call.h"
#endif

namespace visionaray {

#ifdef WITH_CUDA

// ==================================================================
// dynamic array for cuda device data
// ==================================================================

template <typename T>
struct DeviceArray
{
 public:
  typedef T value_type;

  DeviceArray() = default;

  ~DeviceArray()
  {
    CUDA_SAFE_CALL(cudaFree(devicePtr));
    devicePtr = nullptr;
    len = 0;
  }

  DeviceArray(size_t n)
  {
    CUDA_SAFE_CALL(cudaMalloc(&devicePtr, n*sizeof(T)));
    len = n;
  }

  DeviceArray(const DeviceArray &rhs)
  {
    if (rhs != *this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      len = rhs.len;
    }
  }

  DeviceArray(DeviceArray &&rhs)
  {
    if (rhs != *this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(rhs.devicePtr));
      len = rhs.len;
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
  }

  DeviceArray &operator=(const DeviceArray &rhs)
  {
    if (rhs != *this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      len = rhs.len;
    }
    return *this;
  }

  DeviceArray &operator=(DeviceArray &&rhs)
  {
    if (rhs != *this) {
      CUDA_SAFE_CALL(cudaMalloc(&devicePtr, rhs.len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, rhs.devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(rhs.devicePtr));
      rhs.devicePtr = nullptr;
      rhs.len = 0;
    }
    return *this;
  }

  T *data()
  { return devicePtr; }

  const T *data() const
  { return devicePtr; }

  size_t size() const
  { return len; }

  void resize(size_t n)
  {
    if (n == len)
      return;

    T *temp{nullptr};
    if (devicePtr && len > 0) {
      CUDA_SAFE_CALL(cudaMalloc(&temp, len*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(temp, devicePtr, len*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      CUDA_SAFE_CALL(cudaFree(devicePtr));
    }

    CUDA_SAFE_CALL(cudaMalloc(&devicePtr, n*sizeof(T)));

    if (temp) {
      CUDA_SAFE_CALL(cudaMemcpy(devicePtr, temp, std::min(n, len)*sizeof(T),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    len = n;
  }

 private:
  T *devicePtr{nullptr};
  size_t len{0};
};
#endif

// ==================================================================
// host/device array
// ==================================================================

template <typename T>
struct HostDeviceArray : public std::vector<T>
{
 public:
  // TODO: assert trivially copyable
  typedef T value_type;
  typedef std::vector<T> Base;

  HostDeviceArray() = default;
  ~HostDeviceArray() = default;

  void *mapDevice()
  {
    deviceMapped = true;
    return devicePtr();
  }

  void unmapDevice()
  {
    updateOnHost();
    deviceMapped = false;
  }

  void resize(size_t n)
  {
    Base::resize(n);
    updated = true;
  }

  void resize(size_t n, const T &value)
  {
    Base::resize(n, value);
    updated = true;
  }

  void reset(const void *data)
  {
    memcpy(Base::data(), data, Base::size() * sizeof(T));
    updated = true;
  }

  T &operator[](size_t i)
  {
    updated = true;
    return Base::operator[](i);
  }

  const T *hostPtr() const
  {
    return Base::data();
  }

  T *devicePtr()
  {
    updateOnDevice();
    return deviceData.data();
  }

 protected:
#ifdef WITH_CUDA
  DeviceArray<T> deviceData;
#else
  Base deviceData;
#endif
  bool updated = true;
  bool deviceMapped = false;

 private:
  void updateOnDevice()
  {
    if (!updated)
      return;

    deviceData.resize(Base::size());
#ifdef WITH_CUDA
    CUDA_SAFE_CALL(cudaMemcpy(deviceData.data(),
                              Base::data(),
                              Base::size() * sizeof(T),
                              cudaMemcpyHostToDevice));
#else
    memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(T));
#endif
    updated = false;
  }

  void updateOnHost()
  {
    Base::resize(deviceData.size());
#ifdef WITH_CUDA
    CUDA_SAFE_CALL(cudaMemcpy(Base::data(),
                              deviceData.data(),
                              Base::size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
#else
    memcpy(Base::data(), deviceData.data(), Base::size() * sizeof(T));
#endif
    updated = false; // !
  }
};

// ==================================================================
// host/device array storing device object handles
// ==================================================================

struct DeviceHandleArray : public HostDeviceArray<DeviceObjectHandle>
{
 public:
  typedef DeviceObjectHandle value_type;
  typedef HostDeviceArray<DeviceObjectHandle> Base;

  DeviceHandleArray() = default;
  ~DeviceHandleArray() = default;

  void set(size_t index, DeviceObjectHandle handle)
  {
    if (index >= Base::size())
      Base::resize(index+1);

    Base::operator[](index) = handle;
  }
};

// ==================================================================
// Array type capable of managing device-copyable objects via handles
// TODO: can this use HostDeviceArray internally?!
// ==================================================================

template <typename T>
struct DeviceObjectArray : private std::vector<T>
{
 public:
  typedef typename std::vector<T>::value_type value_type;
  typedef std::vector<T> Base;

  DeviceObjectArray() = default;
  ~DeviceObjectArray() = default;

  DeviceObjectHandle alloc(const T &obj)
  {
    Base::push_back(obj);
    updated = true;
    return (DeviceObjectHandle)(Base::size()-1);
  }

  void free(DeviceObjectHandle handle)
  {
    updated = true;
  }

  void update(DeviceObjectHandle handle, const T &obj)
  {
    Base::data()[handle] = obj;
    updated = true;
  }

  size_t size() const
  {
    return Base::size();
  }

  bool empty() const
  {
    return Base::empty();
  }

  void clear()
  {
    Base::clear();
    freeHandles.clear();
    updated = true;
  }

  const T &operator[](size_t index) const
  {
    return Base::operator[](index);
  }

  const T *hostPtr() const
  {
    return Base::data();
  }

  T *devicePtr()
  {
    if (updated) {
      deviceData.resize(Base::size());
#ifdef WITH_CUDA
      CUDA_SAFE_CALL(cudaMemcpy(deviceData.data(),
                     Base::data(),
                     Base::size() * sizeof(T),
                     cudaMemcpyHostToDevice));
#else
      // TODO: assert trivially copyable
      memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(T));
#endif
      updated = false;
    }
    return deviceData.data();
  }

  std::vector<DeviceObjectHandle> freeHandles;
#ifdef WITH_CUDA
  DeviceArray<T> deviceData;
#else
  Base deviceData;
#endif
  bool updated = true;
};

} // namespace visionaray
