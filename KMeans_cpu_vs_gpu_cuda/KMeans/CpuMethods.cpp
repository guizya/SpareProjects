#include "CpuMethods.h"
#include <iostream>

float CPU::calcLength(int pnt, int cluster)
{
	float len = 0;
	for (int dim = 0; dim < D; dim++) {
		float t = (points[pnt * D + dim] - kmeans[cluster * D + dim]);
		len += t * t;
	}
	return len;
}

void CPU::kmeansCPU() {
	float minimal = 1E8;
	int minIdx = -1;
	vector<int> counts(K, 0);
	bool changed = false;
	int loop = 0;

	while (true) {

		// iterating through each sample point to select its cluster
		for (int pnt = 0; pnt < N; pnt++) {

			for (int cluster = 0; cluster < K; cluster++) {
				float length = calcLength(pnt, cluster);

				if (length < minimal) {
					minimal = length;
					minIdx = cluster;
				}
			}

			if (clusters[pnt] != minIdx) {
				changed = true;
				clusters[pnt] = minIdx;
			}

			minimal = 1E8;
			minIdx = -1;
		}

		// if no sample points have changed its cluster, then done
		if (!changed) {
			//cout << "Loop " << loop << endl;
			return;
		}

		// calculate the k mean cluster again
		memset(&kmeans[0], 0, sizeof(float) * K * D);
		for (int pnt = 0; pnt < N; pnt++) {
			int cluster = clusters[pnt];

			counts[cluster] ++;
			for (int dim = 0; dim < D; dim++) {
				kmeans[cluster * D + dim] += points[pnt * D + dim];
			}
		}

		for (int cluster = 0; cluster < K; cluster++) {
			float f = 1.0f / (float)counts[cluster];
			for (int dim = 0; dim < D; dim++) kmeans[cluster * D + dim] *= f;
		}

		//if (K == 32 && loop >= 15) return;

		// preparation for the next loop
		changed = false;
		memset(&counts[0], 0, sizeof(int) * K);
		loop++;
	}
}