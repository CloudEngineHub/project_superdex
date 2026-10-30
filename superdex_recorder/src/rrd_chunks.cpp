/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rrd_chunks.h"

#include <superdex_recorder/formatted_error.h>

#include <mochi_core/utils/span.h>

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/key_value_metadata.h>
#include <lz4.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>

namespace superdex::recorder::detail {
namespace {

static_assert(std::endian::native == std::endian::little, "RRD integers are little-endian.");

using Bytes = mochi::Span<uint8_t const>;

/* The Rerun release whose files this reader decodes: the one the recorder links. Rerun may change
 * any layer of the format in a release, so a Rerun upgrade updates these, and the parity test
 * against Rerun's own reader shows what changed. */
constexpr uint8_t kRerunVersionMajor = 0;
constexpr uint8_t kRerunVersionMinor = 33;
constexpr std::string_view kSorbetVersion = "0.1.3";

// Stream framing, from re_log_encoding::rrd: a header, then messages that each start with their
// kind and length.
constexpr std::string_view kStreamFourcc = "RRF2";
constexpr size_t kStreamHeaderSize = 12;
constexpr size_t kMessageHeaderSize = 16;
constexpr uint64_t kMessageEnd = 0;
constexpr uint64_t kMessageArrowMsg = 2;

// Protobuf enum values, from re_protos.
constexpr uint64_t kStoreKindRecording = 1;
constexpr uint64_t kCompressionLz4 = 2;

std::string_view AsStringView(Bytes bytes) {
  return {reinterpret_cast<char const*>(bytes.data()), bytes.size()};
}

uint64_t ReadU64(Bytes bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

bool ReadVarint(Bytes bytes, size_t& offset, uint64_t& value) {
  value = 0;
  for (uint32_t shift = 0; shift < 64 && offset < bytes.size(); shift += 7) {
    uint8_t const byte = bytes[offset++];
    value |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      return true;
    }
  }
  return false;
}

/* Calls visit(field number, varint, bytes) for each field of a protobuf message. Only varint and
 * length-delimited fields are supported: the messages read here have no others. */
template <typename Visit>
void VisitProtoFields(Bytes message, mochi::Error& error, Visit visit) {
  size_t offset = 0;
  while (error.IsOK() && offset < message.size()) {
    // Both wire types follow the key with a varint: the value, or the length of the bytes.
    uint64_t key = 0;
    uint64_t value = 0;
    bool const valid = ReadVarint(message, offset, key) && ReadVarint(message, offset, value) &&
        ((key & 7) == 0 || ((key & 7) == 2 && value <= message.size() - offset));
    MOCHI_ERROR_IF_NOT(valid, error, "Malformed RRD message.");
    MOCHI_ERROR_RETURN(error);
    if ((key & 7) == 0) {
      visit(key >> 3, value, Bytes{});
    } else {
      visit(key >> 3, 0, message.subspan(offset, value));
      offset += value;
    }
  }
}

struct ArrowMessage {
  uint64_t storeKind = 0;
  std::string_view recordingId;
  uint64_t compression = 0;
  uint64_t uncompressedSize = 0;
  Bytes payload;
};

// rerun.log_msg.v1alpha1.ArrowMsg, with its rerun.common.v1alpha1.StoreId.
ArrowMessage ParseArrowMessage(Bytes bytes, mochi::Error& error) {
  ArrowMessage message;
  VisitProtoFields(bytes, error, [&](uint64_t field, uint64_t varint, Bytes nested) {
    if (field == 1) {
      VisitProtoFields(nested, error, [&](uint64_t storeField, uint64_t storeVarint, Bytes text) {
        if (storeField == 1) {
          message.storeKind = storeVarint;
        } else if (storeField == 2) {
          message.recordingId = AsStringView(text);
        }
      });
    } else if (field == 2) {
      message.compression = varint;
    } else if (field == 3) {
      message.uncompressedSize = varint;
    } else if (field == 5) {
      message.payload = nested;
    }
  });
  return message;
}

/* Returns the payload's Arrow IPC stream in a buffer of its own, aligned as Arrow requires. */
std::shared_ptr<arrow::Buffer> Decompress(ArrowMessage const& message, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  bool const isLz4 = message.compression == kCompressionLz4;
  uint64_t const size = isLz4 ? message.uncompressedSize : message.payload.size();
  constexpr auto kMaxSize = static_cast<uint64_t>(std::numeric_limits<int>::max());
  MOCHI_ERROR_IF(
      size > kMaxSize || message.payload.size() > kMaxSize, error, "RRD chunk is too large.");
  MOCHI_ERROR_RETURN(error, nullptr);
  auto buffer = arrow::AllocateBuffer(static_cast<int64_t>(size));
  MOCHI_ERROR_IF_NOT(buffer.ok(), error, "Failed to allocate an RRD chunk.");
  MOCHI_ERROR_RETURN(error, nullptr);
  auto* const output = reinterpret_cast<char*>((*buffer)->mutable_data());
  auto const* const input = reinterpret_cast<char const*>(message.payload.data());
  if (!isLz4) {
    std::memcpy(output, input, size);
  } else {
    int const decompressedSize = LZ4_decompress_safe(
        input, output, static_cast<int>(message.payload.size()), static_cast<int>(size));
    MOCHI_ERROR_IF(
        decompressedSize != static_cast<int>(size), error, "Failed to decompress an RRD chunk.");
  }
  return std::move(*buffer);
}

std::string GetMetadata(
    std::shared_ptr<arrow::KeyValueMetadata const> const& metadata,
    std::string_view key) {
  return metadata == nullptr ? std::string{} : metadata->Get(key).ValueOr(std::string{});
}

bool IsTimeline(arrow::DataType const& type) {
  // Rerun stores sequences as int64 and durations and timestamps as int64 nanoseconds.
  return type.id() == arrow::Type::INT64 || type.id() == arrow::Type::DURATION ||
      type.id() == arrow::Type::TIMESTAMP;
}

std::optional<RrdChunk> DecodeChunk(std::shared_ptr<arrow::Buffer> ipc, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, std::nullopt);
  auto const decoded = [&]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
    ARROW_ASSIGN_OR_RAISE(
        auto reader,
        arrow::ipc::RecordBatchStreamReader::Open(
            std::make_shared<arrow::io::BufferReader>(std::move(ipc))));
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->Next());
    if (batch == nullptr) {
      return arrow::Status::Invalid("The chunk has no record batch.");
    }
    // Arrow trusts IPC offsets and lengths until they are validated.
    ARROW_RETURN_NOT_OK(batch->ValidateFull());
    return batch;
  }();
  if (!decoded.ok()) {
    SUPERDEX_ERROR_SET(
        error, "Failed to decode an RRD chunk: %s", decoded.status().ToString().c_str());
    return std::nullopt;
  }

  auto const& batch = **decoded;
  auto const& schema = *batch.schema();
  std::string const sorbetVersion = GetMetadata(schema.metadata(), "sorbet:version");
  if (sorbetVersion != kSorbetVersion) {
    SUPERDEX_ERROR_SET(
        error,
        "RRD chunk schema version is '%s', but this reader reads %.*s.",
        sorbetVersion.c_str(),
        static_cast<int>(kSorbetVersion.size()),
        kSorbetVersion.data());
    return std::nullopt;
  }
  std::string entityPath = GetMetadata(schema.metadata(), "rerun:entity_path");
  RrdChunk chunk{.entityPath = entityPath.starts_with('/') ? entityPath.substr(1) : entityPath};
  for (int i = 0; i < schema.num_fields(); ++i) {
    std::string const kind = GetMetadata(schema.field(i)->metadata(), "rerun:kind");
    std::string const& name = schema.field(i)->name();
    auto const& column = batch.column(i);
    if (kind == "index") {
      auto timeline = column->View(arrow::int64());
      if (!IsTimeline(*column->type()) || column->null_count() != 0 || !timeline.ok()) {
        SUPERDEX_ERROR_SET(
            error, "Timeline '%s' of '%s' is not supported.", name.c_str(), entityPath.c_str());
        return std::nullopt;
      }
      chunk.timelines.emplace(name, std::static_pointer_cast<arrow::Int64Array>(*timeline));
    } else if (kind == "data") {
      if (column->type_id() != arrow::Type::LIST) {
        SUPERDEX_ERROR_SET(
            error, "Component '%s' of '%s' is not a list.", name.c_str(), entityPath.c_str());
        return std::nullopt;
      }
      chunk.components.emplace_back(name, std::static_pointer_cast<arrow::ListArray>(column));
    }
  }
  return chunk;
}

std::shared_ptr<arrow::Buffer> MapFile(std::filesystem::path const& path, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  auto const mapped = [&]() -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
    ARROW_ASSIGN_OR_RAISE(
        auto file, arrow::io::MemoryMappedFile::Open(path.string(), arrow::io::FileMode::READ));
    ARROW_ASSIGN_OR_RAISE(int64_t const size, file->GetSize());
    return file->ReadAt(0, size);
  }();
  if (!mapped.ok()) {
    SUPERDEX_ERROR_SET(
        error,
        "Failed to read '%s': %s",
        path.string().c_str(),
        mapped.status().ToString().c_str());
    return nullptr;
  }
  return *mapped;
}

void ValidateStreamHeader(Bytes file, std::string const& path, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (file.size() < kStreamHeaderSize || AsStringView(file.subspan(0, 4)) != kStreamFourcc) {
    SUPERDEX_ERROR_SET(
        error, "'%s' is not an RRD file written by Rerun 0.23 or newer.", path.c_str());
    return;
  }
  uint8_t const* const version = file.data() + 4;
  if (version[0] != kRerunVersionMajor || version[1] != kRerunVersionMinor) {
    SUPERDEX_ERROR_SET(
        error,
        "'%s' was written by Rerun %u.%u.%u, but this reader reads Rerun %u.%u files. "
        "`rerun rrd migrate` upgrades older files.",
        path.c_str(),
        unsigned{version[0]},
        unsigned{version[1]},
        unsigned{version[2]},
        unsigned{kRerunVersionMajor},
        unsigned{kRerunVersionMinor});
  }
}

} // namespace

std::vector<RrdChunk> ReadRrdChunks(std::filesystem::path const& path, mochi::Error& error) {
  auto const buffer = MapFile(path, error);
  MOCHI_ERROR_RETURN(error, {});
  Bytes const file{buffer->data(), static_cast<size_t>(buffer->size())};
  std::string const pathText = path.string();
  ValidateStreamHeader(file, pathText, error);

  std::vector<RrdChunk> chunks;
  std::optional<std::string_view> recordingId;
  size_t offset = kStreamHeaderSize;
  while (error.IsOK()) {
    // A file cut short ends before the end-of-stream message.
    bool const hasHeader = file.size() - offset >= kMessageHeaderSize;
    uint64_t const length = hasHeader ? ReadU64(file, offset + 8) : 0;
    if (!hasHeader || length > file.size() - offset - kMessageHeaderSize) {
      SUPERDEX_ERROR_SET(error, "'%s' was cut short.", pathText.c_str());
      return {};
    }
    uint64_t const kind = ReadU64(file, offset);
    Bytes const message = file.subspan(offset + kMessageHeaderSize, length);
    offset += kMessageHeaderSize + length;

    // The end-of-stream message carries Rerun's chunk index, which is not needed here.
    if (kind == kMessageEnd) {
      return chunks;
    }
    // Other messages describe stores, and blueprint stores configure the viewer.
    ArrowMessage const arrowMessage =
        kind == kMessageArrowMsg ? ParseArrowMessage(message, error) : ArrowMessage{};
    if (!error.IsOK() || arrowMessage.storeKind != kStoreKindRecording) {
      continue;
    }
    if (recordingId.has_value() && *recordingId != arrowMessage.recordingId) {
      SUPERDEX_ERROR_SET(error, "'%s' holds more than one recording.", pathText.c_str());
    }
    recordingId = arrowMessage.recordingId;
    if (auto chunk = DecodeChunk(Decompress(arrowMessage, error), error)) {
      chunks.push_back(*std::move(chunk));
    }
  }
  return {};
}

} // namespace superdex::recorder::detail
