/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_PJRT_IFRT_EXECUTABLE_VERSION_H_
#define XLA_PYTHON_PJRT_IFRT_EXECUTABLE_VERSION_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"

namespace xla {
namespace ifrt {

struct ExecutableVersion : llvm::RTTIExtends<ExecutableVersion, Serializable> {
  ExecutableVersion() = default;
  explicit ExecutableVersion(uint64_t platform_id, std::string runtime_id,
                             std::string runtime_abi_version);

  static absl::StatusOr<std::unique_ptr<ExecutableVersion>> FromProto(
      const ExecutableRuntimeAbiVersion& proto);
  absl::StatusOr<ExecutableRuntimeAbiVersion> ToProto() const;

  // ID that identifies the platform (CPU/GPU/TPU). This corresponds to
  // xla::PjRtPlatformId.
  uint64_t platform_id;
  // ID that identifies the runtime this version is for (e.g. XLA)
  std::string runtime_id;
  // Opaque string that identifies the runtime ABI version.
  std::string runtime_abi_version;

  bool operator==(const ExecutableVersion& other) const;
  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_EXECUTABLE_VERSION_H_
