#pragma once
#include "LatchReg.h"
#include "Memory.h"
#include "PipelineSignal.h"
#include <string>

class Pipeline
{
private:
	LatchReg IF_ID;
	LatchReg ID_EX;
	LatchReg EX_MEM;
	LatchReg MEM_WB;
	PipelineSignal signal;
	unsigned long int pcCount;

public:
	void IF(unsigned long int cycle);
	void ID(unsigned long int cycle);
	void EX(unsigned long int cycle);
	void MEM(unsigned long int cycle);
	void WB(unsigned long int cycle);

	unsigned long int getPCcount();

	//void initialize(std::string filename);

	std::string getString(unsigned long int cycle);

	friend class PipelineSignal;
};