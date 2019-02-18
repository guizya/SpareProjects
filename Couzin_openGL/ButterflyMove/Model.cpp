#include "Model.h"
#include <stdexcept>

int speciesEnable[SPECIES_SIZE];
float speed[SPECIES_SIZE];
int number[SPECIES_SIZE];
float zor[SPECIES_SIZE];
float zoo[SPECIES_SIZE];
float zoa[SPECIES_SIZE];
float blind[SPECIES_SIZE];
float turn[SPECIES_SIZE];
float zooaWeights[SPECIES_SIZE];
float zooaWeightsIntra[SPECIES_SIZE];
float intraWeights[SPECIES_SIZE];
float intraZOR[SPECIES_SIZE];
float intraZOO[SPECIES_SIZE];
float intraZOA[SPECIES_SIZE];

vector< vector<float> > directions;
vector< vector<float> > positions;
vector< vector<float> > nextDirections;

void initialize() {
	directions.resize(SPECIES_SIZE);
	positions.resize(SPECIES_SIZE);
	nextDirections.resize(SPECIES_SIZE);

	speciesEnable[ONE] = 1; speciesEnable[TWO] = 0;
	for (int i = 0; i < SPECIES_SIZE; i++) {
		speed[i] = 0.1;
		number[i] = 10;
		zor[i] = 0.5;
		zoo[i] = 0.5;
		zoa[i] = 0.5;
		blind[i] = 0;
		turn[i] = 75;
		zooaWeights[i] = 0.5;
		zooaWeightsIntra[i] = 0.5;
		intraWeights[i] = 0.5;
		intraZOR[i] = 0.5;
		intraZOO[i] = 0.5;
		intraZOA[i] = 0.5;

		directions[i].resize(number[i] * 3);
		positions[i].resize(number[i] * 3);
		nextDirections[i].resize(number[i] * 3);
		for (int j = 0; j < number[i]; j++) {
			positions[i][j * 3] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
			positions[i][j * 3 + 1] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
			positions[i][j * 3 + 2] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;

			directions[i][j * 3] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			directions[i][j * 3 + 1] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			directions[i][j * 3 + 2] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			Utils::normalize(directions[i][j * 3], directions[i][j * 3 + 1], directions[i][j * 3 + 2]);
		}
	}
}

void reset()
{
	for (int i = 0; i < SPECIES_SIZE; i++) {
		directions[i].resize(number[i] * 3);
		positions[i].resize(number[i] * 3);
		nextDirections[i].resize(number[i] * 3);
		for (int j = 0; j < number[i]; j++) {
			positions[i][j * 3] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
			positions[i][j * 3 + 1] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
			positions[i][j * 3 + 2] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;

			directions[i][j * 3] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			directions[i][j * 3 + 1] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			directions[i][j * 3 + 2] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
			Utils::normalize(directions[i][j * 3], directions[i][j * 3 + 1], directions[i][j * 3 + 2]);
		}
	}
}

void restriction() {
	for (int sp = 0; sp < SPECIES_SIZE; sp++) {
		if (speciesEnable[sp] == 0) continue;

		// first size check
		int size = directions[sp].size();
		if (size != positions[sp].size()) { throw runtime_error("Direction and Position vector size mismatch!"); }
		directions[sp].resize(number[sp] * 3);
		positions[sp].resize(number[sp] * 3);
		nextDirections[sp].resize(number[sp] * 3);

		if (size < number[sp] * 3) {
			for (int j = size / 3; j < number[sp]; j++) {
				positions[sp][j * 3] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
				positions[sp][j * 3 + 1] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;
				positions[sp][j * 3 + 2] = ((float)rand() / (float)RAND_MAX) * borderSize * 2.0f - borderSize;

				directions[sp][j * 3] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
				directions[sp][j * 3 + 1] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
				directions[sp][j * 3 + 2] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
				Utils::normalize(directions[sp][j * 3], directions[sp][j * 3 + 1], directions[sp][j * 3 + 2]);
			}
		}

		// zone length check
		zoo[sp] = (zoo[sp] < zor[sp] ? zor[sp] : zoo[sp]);
		zoa[sp] = (zoa[sp] < zoo[sp] ? zoo[sp] : zoa[sp]);
		intraZOO[sp] = (intraZOO[sp] < intraZOR[sp] ? intraZOR[sp] : intraZOO[sp]);
		intraZOA[sp] = (intraZOA[sp] < intraZOO[sp] ? intraZOO[sp] : intraZOA[sp]);
	}
}

void update(float dT) {

	restriction();

	for (int sp = 0; sp < SPECIES_SIZE; sp++) {
		if (speciesEnable[sp] == 0) continue;

		//cout << "zor " << zor[sp] << " zoo " << zoo[sp] << " zoa " << zoa[sp] << endl;
		for (int i = 0; i < number[sp]; i++) {
			vector<float> dir(3, 0);
			vector<float> pos(3, 0);
			for (int j = 0; j < 3; j++) {
				dir[j] = directions[sp][i * 3 + j];
				pos[j] = positions[sp][i * 3 + j];
			}

			bool repulse0 = false, repulse1 = false;
			vector<float> result(3, 0);
			if (zor[sp] > 0) result = zorCalculation(positions[sp], directions[sp], zor[sp], Utils::degreeToRadian(blind[sp]), i, pos, dir);
			if (!Utils::zeroVector(result)) repulse0 = true;

			if (Utils::zeroVector(result) && !(zoo[sp] == zor[sp] && zoa[sp] == zor[sp])) {
				result = calculation(positions[sp], directions[sp], zoo[sp], zoa[sp], Utils::degreeToRadian(blind[sp]), zooaWeights[sp], i, pos, dir);
			}

			vector<float> result1(3, 0);
			int special = (sp + 1) % SPECIES_SIZE;

			if (speciesEnable[special]) {
				if (intraZOR[sp] > 0) result1 = zorCalculation(positions[special], directions[special], intraZOR[sp], Utils::degreeToRadian(blind[sp]), -1, pos, dir);
				if (!Utils::zeroVector(result1)) repulse1 = true;

				if (Utils::zeroVector(result1) && !(intraZOO[sp] == intraZOR[sp] && intraZOA[sp] == intraZOR[sp])) {
					result1 = calculation(positions[special], directions[special], intraZOO[sp], intraZOA[sp], Utils::degreeToRadian(blind[sp]), zooaWeightsIntra[sp], -1, pos, dir);
				}
			}
			//if (sp == 0) {
			//	cout << "Result " << result[0] << " " << result[1] << " " << result[2] << " Result1 " << result1[0] << " " << result1[1] << " " << result1[2] << " " << intraWeights[sp] << endl;
			//}
			if (repulse0 && !repulse1) {
				// do nothing
			}
			else if (!repulse0 && repulse1) {
				result = result1;
			}
			else {
				if (autoWeight == 1)
					for (int j = 0; j < 3; j++) result[j] = result[j] * (1 - intraWeights[sp]) + result1[j] * intraWeights[sp];
				else
					for (int j = 0; j < 3; j++) result[j] = result[j] * 0.5 + result1[j] * 0.5;
			}
			Utils::normalize(result[0], result[1], result[2]);
			//if (sp == 0) {
			//	cout << "ResultX " << result[0] << " " << result[1] << " " << result[2] << endl;
			//}

			// then turn the direction
			if (Utils::zeroVector(result)) {
				Utils::copy3(nextDirections[sp], i * 3, dir, 0);
			} 
			else {
				if (Utils::includedAngle(dir[0], dir[1], dir[2], result[0], result[1], result[2]) < Utils::degreeToRadian(turn[sp])) {	// this means the agent can turn to the calculated direction
					Utils::copy3(nextDirections[sp], i * 3, result, 0);
				}
				else {
					//cout << "Directions ";
					//for (int m = 0; m < directions[sp].size(); m++) cout << directions[sp][m] << " ";
					//cout << endl;
					//cout << "Positions ";
					//for (int m = 0; m < positions[sp].size(); m++) cout << positions[sp][m] << " ";
					//cout << endl;
					//cout << "result " << result[0] << " " << result[1] << " " << result[2] << endl;
					vector<float> rot = Utils::rotate(Utils::degreeToRadian(turn[sp]), dir[0], dir[1], dir[2], result[0], result[1], result[2]);
					Utils::normalize(rot[0], rot[1], rot[2]);
					//cout << "Rot " << rot[0] << " " << rot[1] << " " << rot[2] << endl;
					Utils::copy3(nextDirections[sp], i * 3, rot, 0);
				}
			}
		}

		// one step over here, and so
		directions[sp].swap(nextDirections[sp]);

		for (int i = 0; i < number[sp]; i++) {
			for (int j = 0; j < 3; j++) {
				float deltaD = directions[sp][i * 3 + j] * speed[sp] * dT;
				
				// boundary check
				if (abs(positions[sp][i * 3 + j] + deltaD) > borderSize) {
					positions[sp][i * 3 + j] -= deltaD;	// out of the boundary, then revert
					directions[sp][i * 3 + j] *= -1.0f;
				} else
					positions[sp][i * 3 + j] += deltaD;
			}
		}
	}
}

vector<float> zorCalculation(const vector<float> &pos, const vector<float> &dir, float d, float bl, int id, const vector<float> &ownPos, const vector<float> &ownDir)
{
	vector<float> result(3, 0);

	vector<float> weights;
	vector<float> immediates;

	//cout << "zor" << endl;
	int size = pos.size() / 3;
	for (int i = 0; i < size; i++) {
		if (i == id) continue;

		vector<float> rIJ (3, 0);
		for (int j = 0; j < 3; j++) rIJ[j] = pos[i * 3 + j] - ownPos[j];

		float len = Utils::length(rIJ[0], rIJ[1], rIJ[2]);
		if (len > d) continue;			// out of ZOR

		float curAngle = Utils::includedAngle(rIJ[0], rIJ[1], rIJ[2], ownDir[0], ownDir[1], ownDir[2]);
		if (curAngle > (Utils::PI - bl)) continue;	// In the blind volume, blind should be half of blind cone
		
		float lenRound = 1.0f / Utils::lengthRound(rIJ[0], rIJ[1], rIJ[2]);
		weights.push_back(autoWeight == 1 ? lenRound : 1.0);	// round the length to avoid 0 lengths

		Utils::normalize(rIJ[0], rIJ[1], rIJ[2]);
		//for (int j = 0; j < 3; j++) result[j] += -rIJ[j];
		for (int j = 0; j < 3; j++) immediates.push_back(-rIJ[j]);
		
	}

	float sum = 0;
	for (int i = 0; i < weights.size(); i++) sum += weights[i];
	for (int i = 0; i < weights.size(); i++) {
		weights[i] /= sum;
		for (int j = 0; j < 3; j++) result[j] += immediates[i * 3 + j] * weights[i];
	}
	Utils::normalize(result[0], result[1], result[2]);

	return result;
}

vector<float> calculation(const vector<float> &pos, const vector<float> &dir, float d1, float d2, float bl, float weight,
	int id, const vector<float> &ownPos, const vector<float> &ownDir)
{
	vector<float> resultZOO(3, 0);
	vector<float> resultZOA(3, 0);
	vector<float> result(3, 0);

	int size = pos.size() / 3;
	bool zooMet = false, zoaMet = false;
	for (int i = 0; i < size; i++) {
		if (i == id) continue;

		vector<float> rIJ(3, 0);
		for (int j = 0; j < 3; j++) rIJ[j] = pos[i * 3 + j] - ownPos[j];

		float len = Utils::length(rIJ[0], rIJ[1], rIJ[2]);
		bool range1 = false, range2 = false;
		if (len <= d1) range1 = true;	// in ZOO
		else if (len <= d2) range2 = true;	// in ZOA
		if (!range1 && !range2) continue;

		float curAngle = Utils::includedAngle(rIJ[0], rIJ[1], rIJ[2], ownDir[0], ownDir[1], ownDir[2]);
		//cout << "Angle " << curAngle << " ref " << (Utils::PI - bl) << endl;
		if (curAngle > (Utils::PI - bl)) continue;	// In the blind volume, blind should be half of blind cone

		if (range2) {
			Utils::normalize(rIJ[0], rIJ[1], rIJ[2]);
			for (int j = 0; j < 3; j++) resultZOA[j] += rIJ[j];
			zoaMet = true;
		}

		if (range1) {
			for (int j = 0; j < 3; j++) resultZOO[j] += dir[i * 3 + j];
			zooMet = true;
		}
	}

	if (zooMet) {
		for (int j = 0; j < 3; j++) resultZOO[j] += ownDir[j];	// count itself
		Utils::normalize(resultZOO[0], resultZOO[1], resultZOO[2]);
	}

	if (zoaMet)
		Utils::normalize(resultZOA[0], resultZOA[1], resultZOA[2]);

	if (zooMet && zoaMet) {
		weight = (autoWeight == 1 ? weight : 0.5f);
		for (int j = 0; j < 3; j++) result[j] = resultZOO[j] * weight + resultZOA[j] * (1.0 - weight);
		Utils::normalize(result[0], result[1], result[2]);
		return result;
	}

	if (zooMet) return resultZOO;
	if (zoaMet) return resultZOA;
	return result;
}

void positionReset()
{
	//for (int i = 0; i < SPECIES_SIZE; i++) {
	int i = 0;
	int num = pow(number[i], 1.0f / 3.0f) + 1.0;
	num = num > 0 ? num : 1;
	float width = 0.1f;
	float deltaWidth = width * 2.0f / num;

	for (int x = 0; x < num; x++) {
		for (int y = 0; y < num; y++) {
			for (int z = 0; z < num; z++) {
				int index = x * num * num + y * num + z;
				if (index >= number[i]) continue;

				positions[i][index * 3] = -width + x * deltaWidth;
				positions[i][index * 3 + 1] = -width + y * deltaWidth;
				positions[i][index * 3 + 2] = -width + z * deltaWidth;

				directions[i][index * 3] = 0;
				directions[i][index * 3 + 1] = -1.0f;
				directions[i][index * 3 + 2] = 0;
			}
		}
	}
	//}
}