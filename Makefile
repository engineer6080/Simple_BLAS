
CXX = g++
CXXFLAGS = -std=c++11 -g
#-ggdb3 for clang related stuff

SOURCES = $(shell find . -name '*.cpp' | sort -k 1nr | cut -f2-)

OBJECTS = $(SOURCES:./%.cpp=./%.o)

EXE = blas

DEPS = $(OBJECTS:.o=.d)

# Libraries used by project
LIBS = -lpthread

#Link the executable
$(EXE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LIBS)
	rm *.o

#Dependencies
-include $(DEPS)

#Compile
./%.o: ./%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sanitize:
	clang -fsanitize=leak $(SOURCES)

tidy:
	clang-tidy $(SOURCES) -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus*

mcheck:
	valgrind --leak-check=full \
	--show-leak-kinds=all \
	--track-origins=yes \
	--verbose \
	--show-reachable=yes \
	--log-file=valgrind-out.txt \
	./$(EXE)

clean:
	-rm -f *.o *.txt $(EXE)
