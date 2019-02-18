#include "Memory.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

char Memory::IMem[2048];
char Memory::DMem[2048];

unsigned long int Memory::readInstructionMemory(unsigned long int pc)
{
	unsigned long int data =
		((IMem[pc] << 24) & 0xff000000) |
		((IMem[pc + 1] << 16) & 0x00ff0000) |
		((IMem[pc + 2] << 8) & 0x0000ff00) |
		(IMem[pc + 3] & 0x000000ff);
	return data;
}

unsigned long int Memory::readDataMemory(unsigned long int address)
{
	unsigned long int data = 
		((DMem[address] << 24) & 0xff000000) |
		((DMem[address + 1] << 16) & 0x00ff0000) |
		((DMem[address + 2] << 8) & 0x0000ff00) |
		(DMem[address + 3] & 0x000000ff);

	return data;
}

void Memory::initialize(std::string filename)
{
	memset(DMem, 0, 2048 * sizeof(char));
	memset(IMem, 0, 2048 * sizeof(char));

	std::ifstream input(filename);
	if (!input.is_open()) {
		throw std::runtime_error("open file error!");
	}

	unsigned int value = 0;
	unsigned long int address = 0;
	unsigned long int mask = 0xff;

	std::string s;
	while (!input.eof()) {

		if (address >= 2048) {
			throw std::runtime_error("Input file is too large!");
		}

		std::getline(input, s);
		std::istringstream digit(s);
		digit >> std::hex >> value;

		IMem[address] = (value >> 24) & mask;
		IMem[address+1] = (value >> 16) & mask;
		IMem[address+2] = (value >> 8) & mask;
		IMem[address+3] = value & mask;
		address += 4;
	}
}

void Memory::writeDataMemory(unsigned long int address, unsigned long int data)
{
	DMem[address] = (data & 0xff000000) >> 24;
	DMem[address + 1] = (data & 0x00ff0000) >> 16;
	DMem[address + 2] = (data & 0x0000ff00) >> 8;
	DMem[address + 3] = (data & 0x000000ff);
}