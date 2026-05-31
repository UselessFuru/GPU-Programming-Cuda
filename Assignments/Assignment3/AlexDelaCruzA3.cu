// Alexander John Dela Cruz and Jose Espino
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// initializes constants
const int arrSize = 8192;       // sample size
const int blocksize = 1024;     // # of threads

__global__
void radix2FFT(float *XR, float *XI) {
	
    	int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    	float evenReal = 0.0f, evenImag = 0.0f;
		float oddReal = 0.0f, oddImag = 0.0f;
	
    	// Sum over even and odd indices
    	for (int i = 0; i < arrSize/2; i++) {
        // even index
        float evenreal = XR[2*i];
        float evenimag = XI[2*i];

        // odd index
        float oddreal  = XR[2*i + 1];
        float oddimag  = XI[2*i + 1];

        float angle = -2.0f * M_PI * globalThreadIdx * i / (arrSize/2); // e^{-j 2pi (k*n)/(N/2)}

        // Rotate even
        evenReal += evenreal * cosf(angle) - evenimag * sinf(angle);
        evenImag += evenreal * sinf(angle) + evenimag * cosf(angle);

        // Rotate odd
        oddReal += oddreal * cosf(angle) - oddimag * sinf(angle);
        oddImag += oddreal * sinf(angle) + oddimag * cosf(angle);
    }

    float twiddleReal = cosf(-2.0f * M_PI * globalThreadIdx / arrSize);
    float twiddleImag = sinf(-2.0f * M_PI * globalThreadIdx / arrSize);

    // Multiply O_k by twiddle factor
    float twiddleOddReal = oddReal * twiddleReal - oddImag * twiddleImag;
    float twiddleOddImag = oddReal * twiddleImag + oddImag * twiddleReal;

    // Compute final X[globalThreadIdx] and X[globalThreadIdx+N/2]
    XR[globalThreadIdx]         = evenReal + twiddleOddReal;
    XI[globalThreadIdx]         = evenImag + twiddleOddImag;
    XR[globalThreadIdx + arrSize/2]   = evenReal - twiddleOddReal;
    XI[globalThreadIdx + arrSize/2]   = evenImag - twiddleOddImag;
}

int main()
{
	// initialize Variables for sol1
	float XR[arrSize] = {0};
 	float XI[arrSize] = {0};
	
	// populate arrays
	XR[0] = 3.6; XI[0] = 2.6;
	XR[1] = 2.9; XI[1] = 6.3;
	XR[2] = 5.6; XI[2] = 4;
	XR[3] = 4.8; XI[3] = 9.1;
	XR[4] = 3.3; XI[4] = 0.4;
	XR[5] = 5.9; XI[5] = 4.8;
	XR[6] = 5; XI[6] = 2.6;
	XR[7] = 4.3; XI[7] = 4.1;
	
	for(int i = 8; i < arrSize ; i++){
		XR[i] = 0; 
		XI[i] = 0; 
	} 
	
	// pointers for arrays 
 	float *XRD, *XID;
	
	
	// gets the size of float data type for
	// moving into the gpu 
 	const float size = arrSize*sizeof(float);

	// Allocates memory of 4 bytes for sol1
 	cudaMalloc( (void**)&XRD, size );
 	cudaMalloc( (void**)&XID, size );


	// copies the values of XR, XI to allocated memory 
	// in the gpu at XRD, XID respectively for sol1
 	cudaMemcpy( XRD, XR, size, cudaMemcpyHostToDevice );
 	cudaMemcpy( XID, XI, size, cudaMemcpyHostToDevice );

	// creates grid with 4 blocks 
 	dim3 dimGridSol1( 4 , 1 , 1 ); 	
	
	// each block will have 1024 threads
	dim3 dimBlockSol1( blocksize, 1 , 1 );

	// runs the kernel radix2FFT
 	radix2FFT<<<dimGridSol1, dimBlockSol1>>>(XRD, XID);
	
	cudaMemcpy(XR, XRD, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(XI, XID, size, cudaMemcpyDeviceToHost);
	
	// frees up memory from pointers XRD, XID,
 	cudaFree( XRD );
	cudaFree( XID );
	
	
	// prints results
	printf("TOTAL PROCESSED SAMPLES: %d\n", arrSize);
    printf("================================\n");
    for (int i = 0; i < 8; i++) {
        printf("XR[%d]: %f XI[%d]: %f \n", i, XR[i], i, XI[i]);
    }
    printf("================================\n");
	
 	return EXIT_SUCCESS;
}
