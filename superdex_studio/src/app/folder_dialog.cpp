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

#include "app/folder_dialog.h"

#include <mochi_core/mochi_platform.h>

#if MOCHI_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// The CMake build force-includes mochi_renderer/windows_compat.h, which includes <windows.h> first
// and then undefines NEAR and FAR (they clash with Filament). commctrl.h, pulled in by shobjidl.h,
// still declares `NEAR*` pointers, so restore the SDK's (empty) definitions around it.
#pragma push_macro("NEAR")
#pragma push_macro("FAR")
#ifndef NEAR
#define NEAR
#endif
#ifndef FAR
#define FAR
#endif
#include <shobjidl.h>
#pragma pop_macro("FAR")
#pragma pop_macro("NEAR")

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")
#endif

namespace superdex::studio {

#if MOCHI_PLATFORM_WINDOWS

namespace {

std::wstring ToWide(std::string const& utf8) {
  if (utf8.empty()) {
    return {};
  }
  int const size =
      MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), size);
  return wide;
}

std::string ToUtf8(wchar_t const* wide) {
  int const size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1) {
    return {};
  }
  std::string utf8(static_cast<size_t>(size - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide, -1, utf8.data(), size, nullptr, nullptr);
  return utf8;
}

} // namespace

std::optional<std::string> ShowNativeFolderDialog(
    char const* title,
    std::string const& defaultPath) {
  // The dialog needs COM on this thread. tinyfiledialogs may already have initialized it in another
  // apartment mode (RPC_E_CHANGED_MODE); the dialog still works then, we just must not
  // uninitialize.
  HRESULT const init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  bool const ownsCom = SUCCEEDED(init);

  std::optional<std::string> result;
  IFileOpenDialog* dialog = nullptr;
  if (SUCCEEDED(CoCreateInstance(
          __uuidof(FileOpenDialog), nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) {
    FILEOPENDIALOGOPTIONS options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    if (title != nullptr) {
      dialog->SetTitle(ToWide(title).c_str());
    }
    if (!defaultPath.empty()) {
      IShellItem* folder = nullptr;
      if (SUCCEEDED(SHCreateItemFromParsingName(
              ToWide(defaultPath).c_str(), nullptr, IID_PPV_ARGS(&folder)))) {
        dialog->SetFolder(folder);
        folder->Release();
      }
    }
    HRESULT const shown = dialog->Show(GetActiveWindow());
    if (SUCCEEDED(shown)) {
      IShellItem* item = nullptr;
      if (SUCCEEDED(dialog->GetResult(&item))) {
        PWSTR path = nullptr;
        if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
          result = ToUtf8(path);
          CoTaskMemFree(path);
        }
        item->Release();
      }
    } else if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
      result = std::string{};
    }
    dialog->Release();
  }

  if (ownsCom) {
    CoUninitialize();
  }
  return result;
}

#else

std::optional<std::string> ShowNativeFolderDialog(
    char const* /*title*/,
    std::string const& /*defaultPath*/) {
  return std::nullopt;
}

#endif

} // namespace superdex::studio
