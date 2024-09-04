# Default target
.PHONY: all
all: run

# Install dependencies
.PHONY: install
install:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run specific script (example)
.PHONY: run
run:
	julia --project=. main.jl

# Clean up any temporary or unwanted files
.PHONY: clean
clean:
	rm *.bson *.txt

