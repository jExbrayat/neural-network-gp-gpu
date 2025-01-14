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
run:
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET) config.json


# For testing purposes, including test.cpp and excluding others
SRCFILES_CPP_TEST = $(filter-out $(SRC_DIR)/autoencoder.cpp $(SRC_DIR)/model.cpp $(SRC_DIR)/main.cpp, $(SRCFILES_CPP))
SRCFILES_CU_TEST = $(filter-out $(SRC_DIR)/gradient_descent.cu, $(SRCFILES_CU))
OBJFILES_CPP_TEST = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCFILES_CPP_TEST))
OBJFILES_CU_TEST = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCFILES_CU_TEST))
OBJFILES_TEST = $(OBJFILES_CPP_TEST) $(OBJFILES_CU_TEST)

# Rule to compile test-related files
test: $(OBJFILES_TEST)
	$(NVCC) -I $(SRC_DIR) -I $(LIBRARIES) $(OBJFILES_TEST) -o $(BUILD_DIR)/test_program

# Clean test-related compiled files
clean_test:
	rm -f $(BUILD_DIR)/test_program $(OBJFILES_TEST)

runtest:
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB $(BUILD_DIR)/test_program
