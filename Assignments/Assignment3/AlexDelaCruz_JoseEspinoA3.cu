// Alexander John Dela Cruz and Jose Espino
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// initializes constants
const int arrSize = 8192;       // sample size (N from the equation) 
const int blocksize = 1024;     // # of threads

__global__
void radix2FFT(float *XR, float *XI) {
	// creates a global thread index
	// cause I'm using more than 1 block
    	int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// initializes float variables for summation later
   	float evenReal = 0.0f, evenImag = 0.0f;
	float oddReal = 0.0f, oddImag = 0.0f;
	
    	// for loop to sum over even and odd indices
    	for (int i = 0; i < arrSize/2; i++) {

        // even index
        float evenreal = XR[2 * i];
        float evenimag = XI[2 * i];

        // odd index
        float oddreal  = XR[(2 * i) + 1];
        float oddimag  = XI[(2 * i) + 1];
	
	// calculates the angle where n (iterator) is i and k (frequency) is globalThreadIdx
	// -i 2pi (k*n)/(N/2) ignoring the i gives -2 pi k * n/ N2
        float angle = -2.0f * M_PI * globalThreadIdx * i / (arrSize/2);

        // calculates summation of even real/imag values 
	// this is essentially the x(2n) cos(2pi (2n)(k) /N) - i sin (2pi (2n)(k)/N) 
        evenReal += (evenreal * cosf(angle)) - (evenimag * sinf(angle));
        evenImag += (evenreal * sinf(angle)) + (evenimag * cosf(angle));

        // calculates summation of odd real/imag values
	// this is essentially the x(2n+1) cos(2pi (2n+1)(k) /N) - i sin (2pi (2n)(k)/N) 
        oddReal += (oddreal * cosf(angle)) - (oddimag * sinf(angle));
        oddImag += (oddreal * sinf(angle)) + (oddimag * cosf(angle));
    }

    // calculates twiddle factor real/imag values
    float twiddleReal = cosf(-2.0f * M_PI * globalThreadIdx / arrSize);
    float twiddleImag = sinf(-2.0f * M_PI * globalThreadIdx / arrSize);

    // multiply odd real/imag by twiddle factor
    float twiddleOddReal = (oddReal * twiddleReal) - (oddImag * twiddleImag);
    float twiddleOddImag = (oddReal * twiddleImag) + (oddImag * twiddleReal);

    // compute final X[globalThreadIdx] and X[globalThreadIdx + (N/2)]
    XR[globalThreadIdx] = evenReal + twiddleOddReal;
    XI[globalThreadIdx] = evenImag + twiddleOddImag;

    XR[globalThreadIdx + (arrSize / 2)]   = evenReal - twiddleOddReal;
    XI[globalThreadIdx + (arrSize / 2)]   = evenImag - twiddleOddImag;
}

int main()
{
	// initialize Variables for XR and XI array for data
	float XR[arrSize] = {0};
 	float XI[arrSize] = {0};
	
	// populate arrays
	XR[0] = 3.6; XI[0] = 2.6;
	XR[1] = 2.9; XI[1] = 6.3;
	XR[2] = 5.6; XI[2] = 4.0;
	XR[3] = 4.8; XI[3] = 9.1;
	XR[4] = 3.3; XI[4] = 0.4;
	XR[5] = 5.9; XI[5] = 4.8;
	XR[6] = 5.0; XI[6] = 2.6;
	XR[7] = 4.3; XI[7] = 4.1;
	
	// 0 padding 
	for(int i = 8; i < arrSize ; i++){
		XR[i] = 0; 
		XI[i] = 0; 
	} 
	
	// pointers for arrays 
 	float *XRD, *XID;
	
	
	// gets the size of float data type for
	// moving into the gpu 
 	const float size = arrSize*sizeof(float);

	// Allocates memory of 8 bytes for sol1
 	cudaMalloc( (void**) &XRD, size );
 	cudaMalloc( (void**) &XID, size );


	// copies the values of XR, XI to allocated memory 
	// in the gpu at XRD, XID respectively for radix2FFT
 	cudaMemcpy(XRD, XR, size, cudaMemcpyHostToDevice);
 	cudaMemcpy(XID, XI, size, cudaMemcpyHostToDevice);

	// creates grid with 4 blocks 
 	dim3 dimGridSol1(4 , 1 , 1); 	
	
	// each block will have 1024 threads
	dim3 dimBlockSol1(blocksize, 1 , 1);

	// runs the kernel radix2FFT
 	radix2FFT<<<dimGridSol1, dimBlockSol1>>>(XRD, XID);
	
	// copies back results from the global memory back (GPU)
	// into the main memory (CPU)
	cudaMemcpy(XR, XRD, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(XI, XID, size, cudaMemcpyDeviceToHost);
	
	// frees up memory from pointers XRD, XID,
 	cudaFree(XRD);
	cudaFree(XID);
	
	
	// prints results
	printf("TOTAL PROCESSED SAMPLES: %d\n", arrSize);
   	printf("================================\n");
   	for (int i = 0; i < 8; i++) {
        printf("XR[%d]: %f XI[%d]: %f \n", i, XR[i], i, XI[i]);
    	}
    	printf("================================\n");
	
 	return EXIT_SUCCESS;
}
