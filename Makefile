# Makefile for CUDA MCTS TicTacToe example

# Compiler and flags
CXX        := clang++
CXXFLAGS   := -std=c++11
CUDAFLAGS  := -x cuda --cuda-gpu-arch=sm_70

# Linker flags
LDFLAGS    := -L/usr/local/cuda/lib64 -lcudart -lcurand

# Target executable
TARGET     := ttt_mcts
SRC        := mcts.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(CUDAFLAGS) $< $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET) 128 1000

clean:
	rm -f $(TARGET)
