#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define SOFTENING 1e-9f
#define COLLISION_DIST 0.01f   // distance threshold for collision

typedef struct { float4 *pos, *vel; } BodySystem;

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(float4 *p, float4 *v, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	// accumulate force components on particle
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

	// Accumulate force components on particle
    for (int tile = 0; tile < gridDim.x; tile++) {
		
		// shared memory array to store the x, y, z positions of a block of particles for fast access
        __shared__ float3 pos_s[BLOCK_SIZE]; 

		int idx = tile * blockDim.x + threadIdx.x;
		
		// load particle from global memory; tpos temporarily holds x, y, z, w components
		float4 tpos = p[idx]; 
		
		// copy only x, y, z into shared memory, discarding w
		pos_s[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z); 
		
		// ensure all threads in the block have finished writing to shared memory before proceeding
		__syncthreads(); 

		// Loop over all particles in this tile
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int global_j = tile * blockDim.x + j;
            if (global_j >= n) break;

            // displacement vector between particles i and j
            float dx = pos_s[j].x - p[i].x;
            float dy = pos_s[j].y - p[i].y;
            float dz = pos_s[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz;

            // collision detection
            if (i < global_j && distSqr < COLLISION_DIST * COLLISION_DIST) {
                printf("Collision: %d & %d -> (%.3f %.3f %.3f) (%.3f %.3f %.3f)\n",
                       i, global_j,
                       p[i].x, p[i].y, p[i].z,
                       pos_s[j].x, pos_s[j].y, pos_s[j].z);
            }

            // N-body gravitational force (with softening)
            float distSqrSoft = distSqr + SOFTENING;
            float invDist = rsqrtf(distSqrSoft);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        __syncthreads();
    }

    v[i].x += dt * Fx;
    v[i].y += dt * Fy;
    v[i].z += dt * Fz;
}

int main(const int argc, const char** argv) {

    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = 2 * nBodies * sizeof(float4);
    float *buf = (float*)malloc(bytes);

    BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

    randomizeBodies(buf, 8 * nBodies);

    float *d_buf;
    cudaMalloc(&d_buf, bytes);

    BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

    int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double totalTime = 0.0;
	
	// record kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
	
    for (int iter = 1; iter <= nIters; iter++) {
        cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
		
		cudaEventRecord(start);
		
        bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		
		float timeComp = 0;
		cudaEventElapsedTime(&timeComp, start, stop); // time in ms
		
		totalTime += timeComp / 1000.0; // converts ms into sec
		
        cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

        // integrate motion
        for (int i = 0; i < nBodies; i++) {
            p.pos[i].x += p.vel[i].x * dt;
            p.pos[i].y += p.vel[i].y * dt;
            p.pos[i].z += p.vel[i].z * dt;
        }
    }
	
    double avgTime = totalTime / (double)(nIters - 1);
    double interactions = 1e-9 * (double)nBodies * (double)nBodies / avgTime;

    printf("\n=== RESULTS ===\n");
    printf("%d Bodies: %.3f Billion Interactions/sec\n", nBodies, interactions);
	printf("Average time per iteration: %.3f sec\n", avgTime);
	printf("Total GPU kernel time: %.3f sec\n", totalTime);

    free(buf);
    cudaFree(d_buf);

    return 0;
}
