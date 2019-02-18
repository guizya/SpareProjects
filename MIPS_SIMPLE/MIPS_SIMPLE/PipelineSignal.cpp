#include "PipelineSignal.h"
#include "Pipeline.h"
#include "Instruction.h"

void PipelineSignal::stopFetching(unsigned long int startCycle)
{
	stopFetch = true;
}

bool PipelineSignal::isStopFetching(unsigned long int cycle)
{
	if (cycle > stopFetchCycle) {
		stopFetch = false;
		stopFetchCycle = 0xffffffff;
	}

	return stopFetch;
}

void PipelineSignal::signalIF(bool branch, unsigned long int address, unsigned long int cycle)
{
	branch2Consume = true;
	this->branch = branch;
	branchAddress = address;
	stopFetchCycle = cycle;
}

void PipelineSignal::branchResolved(bool &branch, unsigned long int &address, unsigned long int cycle)
{
	if (!branch2Consume)  return;

	branch = this->branch;
	address = branchAddress;
	branch2Consume = false;
}

bool PipelineSignal::registerHazard(unsigned long int ra, unsigned long int rb, Pipeline &pipe)
{
	unsigned long int ir1 = pipe.EX_MEM.getIR();
	unsigned long int ir2 = pipe.MEM_WB.getIR();

	unsigned long int rd1 = Instruction::getTargetRegister(ir1);
	unsigned long int rd2 = Instruction::twoTargets(ir1) ? rd1 + 1 : 0;
	unsigned long int rd3 = Instruction::getTargetRegister(ir2);
	unsigned long int rd4 = Instruction::twoTargets(ir2) ? rd3 + 1 : 0;

	if (ra != 0 && (ra == rd1 || ra == rd2 || ra == rd3 || ra == rd4)) return true;
	if (rb != 0 && (rb == rd1 || rb == rd2 || rb == rd3 || rb == rd4)) return true;

	return false;
}

void PipelineSignal::utilization(unsigned long int cycle, float &ifUti, float &idUti, float &exUti, float &memUti, float &wbUti)
{
	ifUti = (float)ifCycle / cycle;
	idUti = (float)idCycle / cycle;
	exUti = (float)exCycle / cycle;
	memUti = (float)memCycle / cycle;
	wbUti = (float)wbCycle / cycle;
}