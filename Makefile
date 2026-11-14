CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
# macOS Apple Clang: default serial build. Linux: opt in with make OPENMP=-fopenmp.
OPENMP ?=
SOEXT := $(shell uname -s | sed 's/Darwin/dylib/;s/Linux/so/')

all: build/libpixelpack.$(SOEXT)
build/libpixelpack.$(SOEXT): src/pixelpack.cpp src/pixelpack.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(OPENMP) -fPIC -shared src/pixelpack.cpp -lz -o $@
	printf '%s\n' '$(CXX) $(CXXFLAGS) $(OPENMP) -fPIC -shared src/pixelpack.cpp -lz' > build/build_command.txt
test: all
	python3 -m unittest discover -s tests -v
sanitize:
	mkdir -p build
	$(CXX) -std=c++17 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/pixelpack.cpp tests/native_test.cpp -lz -o build/native_sanitized
	ASAN_OPTIONS=detect_leaks=0 build/native_sanitized
cuda: all
	nvcc -std=c++17 -O3 cuda/validate.cu -Lbuild -lpixelpack -Xlinker -rpath -Xlinker '$$ORIGIN' -o build/cuda_validate
clean:
	rm -rf build
.PHONY: all test sanitize cuda clean
