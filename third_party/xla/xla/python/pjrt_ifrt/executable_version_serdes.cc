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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/executable_version.h"
#include "xla/tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
namespace {

class ExecutableVersionSerDes
    : public llvm::RTTIExtends<ExecutableVersionSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ExecutableVersion";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for CustomCallProgram serialization"));
    }

    const ExecutableVersion& executable_version =
        llvm::cast<ExecutableVersion>(serializable);

    TF_ASSIGN_OR_RETURN(ExecutableRuntimeAbiVersion executable_version_proto,
                        executable_version.ToProto());
    executable_version_proto.set_version_number(SerDesVersionNumber(0).value());

    return executable_version_proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    ExecutableRuntimeAbiVersion executable_version_proto;
    if (!executable_version_proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse ExecutableVersionProto");
    }
    const SerDesVersionNumber version_number(
        executable_version_proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for ExecutableVersion deserialization"));
    }

    return ExecutableVersion::FromProto(executable_version_proto);
  }

  ExecutableVersionSerDes() = default;
  ~ExecutableVersionSerDes() override = default;

  static char ID;  // NOLINT
};

}  // namespace

[[maybe_unused]] char ExecutableVersionSerDes::ID = 0;

bool register_executable_version_serdes = ([]{
    RegisterSerDes<ExecutableVersion>(
      std::make_unique<ExecutableVersionSerDes>());
}(), true);

}  // namespace ifrt
}  // namespace xla
