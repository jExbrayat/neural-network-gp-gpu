# Compiler and flags
NVCC = nvcc
CXX = g++  # or use your C++ compiler (e.g., clang++ or g++)

# Define directories
SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files: Collect all .cpp and .cu files in the src directory
SRCFILES_CPP = $(wildcard $(SRC_DIR)/*.cpp)
SRCFILES_CU = $(wildcard $(SRC_DIR)/*.cu)

# Included libraries and headers
LIBRARIES = libraries/include

# Object files (compiling .cpp and .cu into .o) -- put them in the build directory
OBJFILES_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCFILES_CPP))
OBJFILES_CU = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCFILES_CU))

# Combine both object files into a single list for the final target
OBJFILES = $(OBJFILES_CPP) $(OBJFILES_CU)

# Rule to compile the program
$(TARGET): $(OBJFILES)
	$(NVCC) -I $(SRC_DIR) -I $(LIBRARIES) $(OBJFILES) -o $(TARGET)

# Rule to create .o intermediary files for .cpp files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -c $< -o $@ -I$(SRC_DIR) -I$(LIBRARIES)

# Rule to create .o intermediary files for .cu files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(LIBRARIES)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET) $(OBJFILES)

# Run rule to execute the program with srun
run: $(TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET) config.json
