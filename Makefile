# Compiler and flags
NVCC = nvcc
CXX = g++  # or use your C++ compiler (e.g., clang++ or g++)

# Define directories
SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main

# Source files: Collect all .cpp files in the src directory, including main.cpp
SRCFILES = $(wildcard $(SRC_DIR)/*.cpp)  # Collect all .cpp files in src

# Included libraries and headers
LIBRARIES = libraries/include

# Object files (compiling .cpp into .o) -- put them in the build directory
OBJFILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCFILES))

# Rule to compile the program
$(TARGET): $(OBJFILES)
	$(NVCC) -I $(SRC_DIR) -I $(LIBRARIES) $(OBJFILES) -o $(TARGET)

# Rule to create .o intermediary files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) -c $< -o $@ -I$(SRC_DIR) -I$(LIBRARIES)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET) $(OBJFILES)

# Run rule to execute the program with srun
run: $(TARGET)
	srun --gres=shard:1 --cpus-per-task=4 --mem=2GB ./$(TARGET) config.json