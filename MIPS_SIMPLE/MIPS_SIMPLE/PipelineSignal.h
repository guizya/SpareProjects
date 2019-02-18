#pragma once

class Pipeline;

class PipelineSignal
{
private:
	bool stopFetch;
	unsigned long int stopFetchCycle;
	bool branch;
	unsigned long int branchAddress;
	bool branch2Consume;

	unsigned long int ifCycle;
	unsigned long int idCycle;
	unsigned long int exCycle;
	unsigned long int memCycle;
	unsigned long int wbCycle;

public:
	PipelineSignal() : stopFetch(false), stopFetchCycle(0xffffffff), branch(false), branchAddress(0), branch2Consume(false),
		ifCycle(0), idCycle(0), exCycle(0), memCycle(0), wbCycle(0)
	{}

	void stopFetching(unsigned long int startCycle); // tell IF to stop fetching
	bool isStopFetching(unsigned long int cycle);

	void signalIF(bool branch, unsigned long int address, unsigned long int cycle);	// signal IF about the resolved branch
	void branchResolved(bool &branch, unsigned long int &address, unsigned long int cycle); // consume the resolved branch, return whether consumed a signal

	bool registerHazard(unsigned long int ra, unsigned long int rb, Pipeline &pipe);

	void countIF() { ifCycle++; };
	void countID() { idCycle++; };
	void countEX() { exCycle++; };
	void countMEM() { memCycle++; };
	void countWB() { wbCycle++; };
	void utilization(unsigned long int cycle, float &ifUti, float &idUti, float &exUti, float &memUti, float &wbUti);
};