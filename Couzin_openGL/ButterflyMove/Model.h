#pragma once
#include <vector>
#include <iostream>

#include "Utils.h"

using namespace std;

enum SPECIES {
	ONE = 0,
	TWO,
	SPECIES_SIZE
};

// suppose the model space is [-border, +border] x [-border, +border] x [-border, +border]
extern float borderSize;
extern float firstSpeed;
extern float deltaT;
extern int autoWeight;

extern int number[SPECIES_SIZE];
extern int speciesEnable[SPECIES_SIZE];
extern float speed[SPECIES_SIZE];
extern float zor[SPECIES_SIZE];
extern float zoo[SPECIES_SIZE];
extern float zoa[SPECIES_SIZE];
extern float blind[SPECIES_SIZE];
extern float turn[SPECIES_SIZE];
extern float zooaWeights[SPECIES_SIZE];
extern float zooaWeightsIntra[SPECIES_SIZE];
extern float intraWeights[SPECIES_SIZE];
extern float intraZOR[SPECIES_SIZE];
extern float intraZOO[SPECIES_SIZE];
extern float intraZOA[SPECIES_SIZE];

extern vector< vector<float> > directions;
extern vector< vector<float> > positions;

extern void initialize();
extern void reset();
extern void positionReset();
extern void update(float dT);
extern vector<float> zorCalculation(const vector<float> &pos, const vector<float> &dir, float d, float blind, int id, const vector<float> &ownPos, const vector<float> &ownDir);
extern vector<float> calculation(const vector<float> &pos, const vector<float> &dir, float d1, float d2, float blind, float weight,
								 int id, const vector<float> &ownPos, const vector<float> &ownDir);