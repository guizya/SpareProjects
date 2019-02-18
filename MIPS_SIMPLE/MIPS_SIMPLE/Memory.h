#pragma once
#include <string>

class Memory
{
private:
	static char IMem[2048];
	static char DMem[2048];

public:
	static unsigned long int readInstructionMemory(unsigned long int pc);
	static unsigned long int readDataMemory(unsigned long int address);
	static void initialize(std::string filename);
	static void writeDataMemory(unsigned long int address, unsigned long int data);
};