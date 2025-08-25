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

#include "xla/python/pjrt_ifrt/executable_version.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"

namespace xla {
namespace ifrt {
[[maybe_unused]] char ExecutableVersion::ID = 0;

ExecutableVersion::ExecutableVersion(uint64_t platform_id,
                                     std::string runtime_id,
                                     std::string runtime_abi_version)
    : platform_id(platform_id),
      runtime_id(std::move(runtime_id)),
      runtime_abi_version(std::move(runtime_abi_version)) {}

absl::StatusOr<std::unique_ptr<ExecutableVersion>> ExecutableVersion::FromProto(
    const ExecutableRuntimeAbiVersion& proto) {
  return std::make_unique<ExecutableVersion>(
      proto.platform_id(), proto.runtime_id(), proto.runtime_abi_version());
}

absl::StatusOr<ExecutableRuntimeAbiVersion> ExecutableVersion::ToProto() const {
  ExecutableRuntimeAbiVersion proto;
  proto.set_platform_id(platform_id);
  proto.set_runtime_id(runtime_id);
  proto.set_runtime_abi_version(runtime_abi_version);
  return proto;
}

bool ExecutableVersion::operator==(const ExecutableVersion& other) const {
  return platform_id == other.platform_id && runtime_id == other.runtime_id &&
         runtime_abi_version == other.runtime_abi_version;
}

}  // namespace ifrt
}  // namespace xla
