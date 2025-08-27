CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

TARGET = main_pipeline
SOURCE = main_pipeline.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)
	rm -rf output1/

run_denver: $(TARGET)
	./$(TARGET) CSVs1/Denver.csv

run_washington: $(TARGET)
	./$(TARGET) CSVs1/Wahington.csv

run_sanfrancisco: $(TARGET)
	./$(TARGET) CSVs1/San\ Francisco.csv

run_seattle: $(TARGET)
	./$(TARGET) CSVs1/Seattle.csv

run_lasvegas: $(TARGET)
	./$(TARGET) CSVs1/Las\ Vegas.csv

# Run all cities
run_all: run_denver run_washington run_sanfrancisco run_seattle run_lasvegas
	@echo "All cities processed successfully!"

openmp_mosp: OpenMP_MOSP.cpp
	clang++ -std=c++17 -Wall -O2 -Xclang -fopenmp -L$(shell brew --prefix libomp)/lib -lomp -o openmp_mosp OpenMP_MOSP.cpp

.PHONY: all clean run_denver run_washington run_sanfrancisco run_seattle run_lasvegas run_all 