CCFLAG := -w -O3 -arch=sm_20 -lpthread 
NAME := main

all: $(NAME)

$(NAME): $(NAME).cu $(SHARED)
	nvcc $(NAME).cu $(CCFLAG) -o $(NAME)
clean: 
	rm -f $(NAME)
	rm -f *.o
