// Alexander John Dela Cruz 
#include <stdio.h>
#include <stdlib.h>

// initializes constants
const int arrSize = 10240;
const int blocksize = 1024;

// kernel for sol1 matrix product (non-cyclic partition)
__global__
void matrixProductSol1(int *a1, int *b1, int *c1){
	// create variables to keep track of when each block 
	// begins and ends by dividing total array with the 
	// number of blocks available 
	int blockBegin = blockIdx.x * (arrSize / gridDim.x);
	int blockEnd = blockBegin + (arrSize / gridDim.x);

	// each thread handles several elements in its block chunk
	// and jumps ahead by blockDim.x each time
	// this ensures threads dont do the same work
	for(int i = blockBegin + threadIdx.x; i < blockEnd; i += blockDim.x){
 	c1[i] = a1[i] * b1[i];
	}
}
	// kernel for Sol2 matrix product (cyclic partition)
__global__
void matrixProductSol2(int *a2, int *b2, int *c2){
	// global thread index variable to keep track of threads
	int globalThreadIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// starts at the globalThreadIdx to keep track of individual thread
	// then increments by blockDim.x * gridDim.x (total threads in all blocks) 
	// so that it covers the array cyclically without any overlaps
	for(int i = globalThreadIdx; i < arrSize; i += blockDim.x * gridDim.x){
	c2[i] = a2[i] * b2[i];
	}
}

	// kernel for Sol3 matrix product 10 blocks max threads
__global__
void matrixProductSol3(int *a3, int *b3, int *c3){
	// global thread index variable to keep track of threads
	int globalThreadIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// calculates the product of array a and array b and puts it 
	// in array c 
	c3[globalThreadIdx] = a3[globalThreadIdx] * b3[globalThreadIdx];
}

int main()
{
	/* i know using 3 different variables for the kernels isnt very
	   ideal and that i could reuse them, but i wanted to make
	   sure that its that specific kernels output
	*/
	// initialize Variables for sol1
	int a1[arrSize] = {0};
 	int b1[arrSize] = {0};
	int c1[arrSize] = {0};
	
	// initializes Variables for sol2
	int a2[arrSize] = {0};
	int b2[arrSize] = {0};
	int c2[arrSize] = {0};
	
	// initializes Variables for sol3
	int a3[arrSize] = {0};
	int b3[arrSize] = {0};
	int c3[arrSize] = {0};
	
	// populate arrays
	for(int i = 0; i < arrSize ; i++){
		// for Sol1
		a1[i] = 2 * i; 
		b1[i] = (2 * i) + 1; 
		// for Sol2
		a2[i] = 2 * i; 
		b2[i] = (2 * i) + 1; 
		// for Sol3
		a3[i] = 2 * i; 
		b3[i] = (2 * i) + 1; 
		
	} 
	
	// pointers for arrays 
 	int *ad1, *ad2, *ad3;
 	int *bd1, *bd2, *bd3;
	int *cd1, *cd2, *cd3;
	
	
	// gets the size of int data type for
	// moving into the gpu 
 	const int size = arrSize*sizeof(int);

	// Allocates memory of 4 bytes for sol1
 	cudaMalloc( (void**)&ad1, size );
 	cudaMalloc( (void**)&bd1, size );
	cudaMalloc( (void**)&cd1, size );
	
	// Allocates memory for 4 bytes for sol2
	cudaMalloc( (void**)&ad2, size );
 	cudaMalloc( (void**)&bd2, size );
	cudaMalloc( (void**)&cd2, size );
	
	// Allocates memory for 4 bytes for sol3
	cudaMalloc( (void**)&ad3, size );
 	cudaMalloc( (void**)&bd3, size );
	cudaMalloc( (void**)&cd3, size );

	// copies the values of a, b,c to allocated memory 
	// in the gpu at ad, bd, cd respectively for sol1
 	cudaMemcpy( ad1, a1, size, cudaMemcpyHostToDevice );
 	cudaMemcpy( bd1, b1, size, cudaMemcpyHostToDevice );
	cudaMemcpy( cd1, c1, size, cudaMemcpyHostToDevice );
	
	// copies the values of a, b,c to allocated memory 
	// in the gpu at ad, bd, cd respectively for sol2
	cudaMemcpy( ad2, a2, size, cudaMemcpyHostToDevice );
 	cudaMemcpy( bd2, b2, size, cudaMemcpyHostToDevice );
	cudaMemcpy( cd2, c2, size, cudaMemcpyHostToDevice );
	
	// copies the values of a, b,c to allocated memory 
	// in the gpu at ad, bd, cd respectively for sol3
 	cudaMemcpy( ad3, a3, size, cudaMemcpyHostToDevice );
 	cudaMemcpy( bd3, b3, size, cudaMemcpyHostToDevice );
	cudaMemcpy( cd3, c3, size, cudaMemcpyHostToDevice );
	

	// creates a grid with 2 blocks for sol1 and sol2
 	dim3 dimGridSol1( 2 , 1 , 1 ); 	
	dim3 dimGridSol2( 2 , 1 , 1 );
	
	// creates grid with 10 blocks for sol3
	dim3 dimGridSol3( 10, 1, 1 );
	
	// each block will have 1024 threads
	dim3 dimBlockSol1( blocksize, 1 , 1 );
	dim3 dimBlockSol2( blocksize, 1 , 1 );
	dim3 dimBlockSol3( blocksize, 1 , 1 );

	// runs the kernel matrixProduct 
	// which multiplies the array ad with bd
	// and puts it in array cd 
 	matrixProductSol1<<<dimGridSol1, dimBlockSol1>>>(ad1, bd1, cd1);
	
	matrixProductSol2<<<dimGridSol1, dimBlockSol1>>>(ad2, bd2, cd2);
	
	matrixProductSol3<<<dimGridSol3, dimBlockSol3>>>(ad3, bd3, cd3);

	// allocates data from array cd back into the host
 	cudaMemcpy( c1, cd1, size, cudaMemcpyDeviceToHost );
	cudaMemcpy( c2, cd2, size, cudaMemcpyDeviceToHost );
	cudaMemcpy( c3, cd3, size, cudaMemcpyDeviceToHost );
	
	
	// frees up memory from pointers ad, bd, cd
 	cudaFree( ad1 );
	cudaFree( bd1 );
	cudaFree( cd1 ); 
	
	cudaFree( ad2 );
	cudaFree( bd2 );
	cudaFree( cd2 ); 
	
	cudaFree( ad3 );
	cudaFree( bd3 );
	cudaFree( cd3 ); 
	
	// prints results
	printf("2 Blocks - Not Cyclic ( C[0], C[10239] ) = (%d, %d)\n", c1[0], c1[arrSize-1]);
	printf("2 Blocks - Cyclic ( C[0], C[10239] ) = (%d, %d)\n", c2[0], c2[arrSize-1]);
	printf("10 Blocks - ( C[0], C[10239] ) = (%d, %d)\n", c3[0], c3[arrSize-1]);
	
	
 	return EXIT_SUCCESS;
}
