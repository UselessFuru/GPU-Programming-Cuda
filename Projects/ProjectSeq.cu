#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer2.h"

#define SOFTENING 1e-9f
#define COLLISION_DIST 0.01f   // distance threshold for collision

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
	 
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      
	  // collision detection
            if (i < j && distSqr < COLLISION_DIST * COLLISION_DIST) {
                printf("Collision: %d & %d -> (%.3f %.3f %.3f) (%.3f %.3f %.3f)\n",
                       i, j,
                       p[i].x, p[i].y, p[i].z,
                       p[j].x, p[j].y, p[j].z);
            }
			
	  // Gravitational force with softening
      float distSqrSoft = distSqr + SOFTENING;
	  float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx; 
    p[i].vy += dt * Fy; 
    p[i].vz += dt * Fz;
  }
}

int main(const int argc, const char** argv) {

    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(Body);
    float *buf = (float*)malloc(bytes);
    Body *p = (Body*)buf;

    randomizeBodies(buf, 6*nBodies);

    double start, finish, elapsed;
    double totalTime = 0.0;

    for (int iter = 1; iter <= nIters; iter++) {
        GET_TIME(start);

        bodyForce(p, dt, nBodies);

        for (int i = 0; i < nBodies; i++) {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }

        GET_TIME(finish);
        elapsed = finish - start;

        totalTime += elapsed; 

   
    }

  double avgTime = totalTime / (double)(nIters - 1); 
  double interactions = 1e-9 * (double)nBodies * (double)nBodies / avgTime;

  printf("\n=== RESULTS ===\n");
  printf("%d Bodies: %.3f Billion Interactions/sec\n", nBodies, interactions);
  printf("Average time per iteration (excluding warm-up): %.3f sec\n", avgTime);
  printf("Total time for all iterations: %.3f sec\n", totalTime);
  
  free(buf);
  return 0;
}
