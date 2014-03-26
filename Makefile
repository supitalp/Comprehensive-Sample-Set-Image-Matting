.PHONY: clean, mrproper
.SUFFIXES:


CXX = g++
CXXFLAGS = -O3 -Wall

LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml
LIBS = -L/usr/local/lib
INC = -I/usr/local/include/opencv

all : CSSMatting.o Region.o CandidateSample.o
	$(CXX) $^ -o cssmatting $(CXXFLAGS) $(INC) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC) $(LIBS) $(LDFLAGS)

clean:
	rm -rf *.o

mrproper: clean
	rm -rf cssmatting
