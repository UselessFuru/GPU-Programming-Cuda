// Alexander John Dela Cruz 
#include <stdio.h>
#include <stdlib.h>

const int arrSize = 4096;
const int blocksize = 64;

__global__
void matrixProduct(int *a, int *b, int *c)
{
	// Creates a global thread index
	// cause I'm using more than 1 block
	int globalThreadIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// stores product of a and b into c
 	c[globalThreadIdx] = a[globalThreadIdx] * b[globalThreadIdx];
}

int main()
{
	// initialize Variables (CPU)
	int a[arrSize] = {0};
 	int b[arrSize] = {0};
	int c[arrSize] = {0};
	
	
	// populate arrays (CPU)
	for(int i = 0; i < arrSize ; i++){
		a[i] = i; 
		b[i] = 4095 + i; 
	} 
	
	// pointers for arrays 
 	int *ad;
 	int *bd;
	int *cd;
	
	// gets the size of int data type for
	// moving into the gpu 
 	const int size = arrSize*sizeof(int);

	// Allocates memory to the device (GPU)
 	cudaMalloc( (void**)&ad, size );
 	cudaMalloc( (void**)&bd, size );
	cudaMalloc( (void**)&cd, size );

	// copies the values of a, b to allocated memory 
	// since gpu cant access cpu memory (GPU)
 	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );
 	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );

	// creates a grid with 64 blocks because 4096 / 64 = 64 blocks
 	dim3 dimGrid( 64 ,1, 1 ); 	
	
	// each block will have 64 threads 64 blocks with 64 threads
	// 64 * 64 = 4096 which means every thread is doing 1 thing
	dim3 dimBlock( blocksize, 1 , 1);

	// runs the kernel matrixProduct 
	// which multiplies the array ad with bd
	// and puts it in array cd 
 	matrixProduct<<<dimGrid, dimBlock>>>(ad, bd, cd);

	// allocates data from array cd back into the host
 	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
	
	// frees up memory from pointers ad, bd, cd
 	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd ); 

	// prints results
	printf("Summary Results of Array C:\n");
	printf("[%d]: %d * %d = %d \n", 0, a[0], b[0], c[0]);
	printf("[%d]: %d * %d = %d \n", 4095, a[4095], b[4095], c[4095]);

	// prints Summation of array C
	int sum = 0;
	for(int i = 0; i < arrSize; i++){
		sum += c[i];
	}
	printf("Summation of array C: %d", sum);
	
 	return EXIT_SUCCESS;
}
