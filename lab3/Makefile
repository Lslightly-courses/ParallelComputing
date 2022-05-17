all: serial MPISpMV

serial:
	@g++ -DSERIAL util.cc SpMV.cc -o ./serialSpMV

MPISpMV:
	@mpicxx util.cc SpMV.cc -o ./parallelSpMV


DEBUGMPISpMV:
	@mpicxx -DDEBUG util.cc SpMV.cc -o ./parallelSpMV
