# 6. 빌드 시스템과 vcpkg

## 6.1 ggml의 CMake 구조

```
CMakeLists.txt              옵션 정의(GGML_CUDA, GGML_NATIVE, GGML_BACKEND_DL ...), 표준 설정(C11/C++17), 설치 규칙, ggml-config.cmake 생성
src/CMakeLists.txt          ggml-base / ggml 타깃, ggml_add_backend_library(), 백엔드 18종 등록
src/ggml-cpu/CMakeLists.txt CPU 변종(ggml_add_cpu_backend_variant), ISA 플래그, OpenMP/Accelerate/KleidiAI
src/ggml-<backend>/CMakeLists.txt 각 백엔드
cmake/common.cmake          경고 플래그, 아키텍처 감지(ggml_get_system_arch)
cmake/ggml-config.cmake.in  find_package(ggml) 소비자용 설정 (백엔드별 find_dependency)
ggml.pc.in                  pkg-config
examples/test-cmake/        설치된 ggml을 find_package로 쓰는 소비자 예제
```

소비자 쪽은 이렇게 쓴다(`examples/test-cmake/CMakeLists.txt`):
```cmake
find_package(ggml 0.19.0 REQUIRED)
target_link_libraries(test-cmake PRIVATE ggml::ggml)
```

`ggml-config.cmake.in`은 정적 빌드일 때 CUDA/Metal/Vulkan/OpenMP 등 **모든 백엔드의 전이 의존성**을 `find_dependency`로 다시 찾는다(`cmake/ggml-config.cmake.in:7-110`). 백엔드가 늘수록 이 파일이 비대해진다. 새 프로젝트에서는 백엔드를 **별도 CMake 컴포넌트**(`find_package(x COMPONENTS cuda)`)로 나누는 편이 vcpkg feature와 1:1로 대응돼 관리가 쉽다.

주요 옵션(`CMakeLists.txt:86-218`):
| 옵션 | 기본 | 의미 |
|---|---|---|
| `GGML_NATIVE` | ON(호스트 빌드) | `-march=native`. 배포 빌드에서는 OFF |
| `GGML_BACKEND_DL` | OFF | 백엔드를 플러그인(.so/.dll)으로 |
| `GGML_CPU_ALL_VARIANTS` | OFF | ISA별 CPU 백엔드 여러 개 빌드 (DL 필요) |
| `GGML_STATIC` | OFF | 정적 링크 |
| `GGML_LTO` | OFF | IPO |
| `GGML_SANITIZE_*` | OFF | ASan/TSan/UBSan |
| `GGML_CPU_REPACK` | ON | Q4_0 런타임 재배치 |
| `GGML_OPENMP` | ON | OpenMP 스레딩 |
| `GGML_CUDA`, `GGML_VULKAN`, `GGML_METAL` ... | OFF | 백엔드 |

## 6.2 vcpkg 포트가 ggml을 어떻게 다루는가

(0장 표에서 본 `vcpkg.json` 참조)

portfile의 처리 순서:
1. `vcpkg_from_github(REPO ggml-org/ggml, REF v${VERSION})` + 패치 적용.
2. feature → CMake 옵션 매핑: `blas→GGML_BLAS`, `cuda→GGML_CUDA`, `metal→GGML_METAL`, `opencl→GGML_OPENCL`(Python3 필요), `openmp→GGML_OPENMP`, `vulkan→GGML_VULKAN`(호스트 빌드 셰이더 도구 경로 전달).
3. `GGML_NATIVE=OFF`, `GGML_AVX/AVX2/FMA/F16C/BMI2/SSE42=OFF` 등 명시적으로 끔. `GGML_SYCL`, `GGML_HIP`도 OFF.
4. ARM64 MSVC에서는 CPU 백엔드 비활성.
5. `vcpkg_cmake_config_fixup()`으로 `ggml-config.cmake` 경로 교정, 동적 링크용 헤더 수정, 디버그 헤더/문서 제거, 라이선스 설치.

배울 점:
- **feature = CMake 옵션 = 링크 의존성** 세 가지가 정확히 대응돼야 포트가 단순해진다.
- 포트가 상류를 패치해야 했다는 것은 상류 CMake가 "설치 후 소비"를 완전히 지원하지 않았다는 뜻이다. 새 프로젝트는 `install(EXPORT)` + `write_basic_package_version_file` + 소비자 테스트(`examples/test-cmake` 같은)를 처음부터 CI에 넣는다.
- 네이티브 최적화는 **삼중항(triplet)** 이나 별도 feature(`native`)로 사용자에게 맡긴다.

## 6.3 새 프로젝트의 vcpkg 구성 (권장 뼈대)

### 소비자로서: 매니페스트 모드

```json
// vcpkg.json
{
  "name": "ggml23",
  "version-semver": "0.1.0",
  "description": "ggml re-implementation in pure C++23",
  "license": "MIT",
  "dependencies": [],
  "default-features": [],
  "features": {
    "tests": {
      "description": "Build differential tests against upstream ggml",
      "dependencies": [ "catch2", "ggml" ]
    },
    "highway": {
      "description": "Use Google Highway for portable SIMD kernels",
      "dependencies": [ "highway" ]
    },
    "vulkan": {
      "description": "Vulkan backend",
      "dependencies": [ "vulkan", "spirv-headers", {"name": "shaderc", "host": true} ]
    }
  },
  "builtin-baseline": "<vcpkg 커밋 해시>"
}
```

- 코어는 **의존성 0개**를 유지한다(ggml의 "no dependencies" 철학과 동일). 이는 vcpkg 포트 심사에서도 유리하다.
- `ggml`을 `tests` feature에만 넣어 **차등 테스트 오라클**로 쓴다.
- `builtin-baseline`으로 버전을 고정한다. 재현 가능한 빌드의 핵심.

```json
// vcpkg-configuration.json  (자체 레지스트리/오버레이가 필요할 때)
{
  "default-registry": { "kind": "git", "repository": "https://github.com/microsoft/vcpkg", "baseline": "<해시>" },
  "overlay-ports": [ "./ports" ]
}
```

```json
// CMakePresets.json (발췌)
{
  "version": 6,
  "configurePresets": [
    {
      "name": "vcpkg",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": { "CMAKE_CXX_STANDARD": "23", "CMAKE_CXX_STANDARD_REQUIRED": "ON", "CMAKE_CXX_EXTENSIONS": "OFF" }
    },
    { "name": "vcpkg-tests", "inherits": "vcpkg", "cacheVariables": { "VCPKG_MANIFEST_FEATURES": "tests" } }
  ]
}
```

### 생산자로서: 포트 만들기

레지스트리 등록 전에 `ports/ggml23/` 오버레이 포트를 저장소 안에 두고 CI에서 `vcpkg install ggml23 --overlay-ports=./ports`로 검증한다. 포트에 필요한 것:
- `vcpkg.json`: 위와 동일하되 `"dependencies"`에 `vcpkg-cmake`, `vcpkg-cmake-config`(host).
- `portfile.cmake`: `vcpkg_from_github` → `vcpkg_cmake_configure(FEATURE_OPTIONS ...)` → `vcpkg_cmake_install` → `vcpkg_cmake_config_fixup(PACKAGE_NAME ggml23)` → `vcpkg_install_copyright`.
- `usage`: 소비자 안내문.

vcpkg 유지관리자 가이드가 요구하는 것: 기능은 기본 OFF, 헤더 전용이면 `vcpkg_cmake_config_fixup` 뒤 `lib` 디렉터리 제거, 정적/동적 모두 빌드 가능, 디버그 빌드에서 include 중복 제거. `examples/test-cmake` 스타일의 소비자 테스트를 포트 CI(`vcpkg ci` 대신 매니페스트로 전체 설치)로 돌린다.

### 삼중항 결정

| 삼중항 | C++23 상태(2026) | 비고 |
|---|---|---|
| `x64-linux` (GCC 14+/Clang 18+) | 언어 완전, `std::mdspan`은 GCC 15 / libc++ 최신 필요 | `import std`는 실험적 |
| `x64-windows` (MSVC 17.10+) | 언어 완전, `std::mdspan`/`std::expected`/`std::print` 지원, **`import std` 가장 앞섬** | `std::float16_t` 없음 |
| `arm64-osx` (AppleClang) | libc++ 버전에 따라 `std::mdspan`, `std::expected` 차이 | 최신 Xcode 필요 |

`std::mdspan`이 없으면 vcpkg의 `mdspan`(kokkos 참조 구현) 포트를 폴리필로 쓸 수 있다. 이 경우 `#if __cpp_lib_mdspan` 분기를 코어에 두어야 한다.

## 6.4 ggml 상류 옵션 중 새 프로젝트가 상속해야 할 것

- **`GGML_BACKEND_DL` 모델**: 플러그인 경계를 C ABI로 유지. C++23 모듈이나 클래스는 경계 안쪽에서만.
- **`GGML_CPU_ALL_VARIANTS` 모델**: 배포 바이너리는 범용 + 런타임 ISA 선택. Highway를 쓰면 이 부분이 라이브러리 수준에서 해결된다.
- **`GGML_SANITIZE_*`**: CI 프리셋으로 유지.
- **`GGML_FATAL_WARNINGS` + `-Wall -Wextra -Wpedantic -Wcast-qual`**(`src/CMakeLists.txt:39`): C++23에서는 `-Wshadow`, `-Wconversion`까지 올릴 수 있다.

## 이 장의 체크포인트

- [ ] `find_package(ggml)`가 백엔드 의존성을 어떻게 다시 찾는지 설명할 수 있다.
- [ ] vcpkg 포트가 `GGML_NATIVE`를 끄는 이유를 말할 수 있다.
- [ ] 새 프로젝트의 `vcpkg.json`에서 `ggml`이 코어 의존성이 아니라 `tests` feature에 있는 이유를 설명할 수 있다.
