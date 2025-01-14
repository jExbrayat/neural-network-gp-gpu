# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I"libraries/include" -I"src"
NVCCFLAGS = -std=c++17 -I"libraries/include" -I"src"

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
CUDA_SRCS = $(SRC_DIR)/cuda_operations.cu $(SRC_DIR)/gradient_descent.cu $(SRC_DIR)/cuda_utils.cu $(SRC_DIR)/test.cu
CPP_SRCS = $(filter-out $(SRC_DIR)/main.cpp, $(wildcard $(SRC_DIR)/*.cpp))
MAIN_CPP = $(SRC_DIR)/main.cpp
TEST_CPP = $(SRC_DIR)/test.cu

# Object files
CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))
MAIN_OBJ = $(BUILD_DIR)/main.o
TEST_OBJ = $(BUILD_DIR)/test.o

# Executables
MAIN_EXEC = build/main_program
TEST_EXEC = build/test_program

# Rules
.PHONY: all clean

all: $(MAIN_EXEC) $(TEST_EXEC)

# Rule for the main program
$(MAIN_EXEC): $(filter-out $(TEST_OBJ), $(CUDA_OBJS) $(CPP_OBJS) $(MAIN_OBJ))
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Rule for the test program
$(TEST_EXEC): $(filter-out $(BUILD_DIR)/model.o $(BUILD_DIR)/autoencoder.o $(BUILD_DIR)/gradient_descent.o $(MAIN_OBJ), $(CPP_OBJS) $(CUDA_OBJS)) $(TEST_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile CUDA source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run main program
run:
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB $(MAIN_EXEC) config.json

# Run test program
runtest:
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB $(TEST_EXEC)


# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(MAIN_EXEC) $(TEST_EXEC)