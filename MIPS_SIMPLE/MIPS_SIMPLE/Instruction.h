#pragma once

#include <vector>

enum Op {
	SLL = 0,
	SRL = 2,
	BEQ = 4,
	ADDI = 8,
	SLTI = 10,
	SLTIU = 11,
	ANDI = 12,
	ORI = 13,
	LUI = 15,
	MUL = 24,
	ADD = 32,
	SUB = 34,
	LW = 35,
	AND = 36,
	OR = 37,
	SW = 43,
};

class Instruction
{
private:
	static unsigned long int special(unsigned long int ir);
	static unsigned long int rs(unsigned long int ir);
	static unsigned long int rt(unsigned long int ir);
	static unsigned long int rd(unsigned long int ir);
	static unsigned long int opcode(unsigned long int ir);
	static unsigned long int immediate(unsigned long int ir);
	static unsigned long int sa(unsigned long int ir);
	static unsigned long int signExtend(unsigned long int imm);
	static unsigned long int zeroExtend(unsigned long int imm);
	static bool check_overflow(unsigned long long int value);
public:
	static bool isALU(unsigned long int ir);
	static bool isBRA(unsigned long int ir);
	static bool isLoadStore(unsigned long int ir);
	static bool isLoad(unsigned long int ir);
	static bool isStore(unsigned long int ir);
	static bool isNop(unsigned long int ir);
	static bool hasImm(unsigned long int ir);
	static unsigned long int getImm(unsigned long int ir);
	static unsigned long int getTargetRegister(unsigned long int ir);
	static void getSourceRegisters(unsigned long int ir, unsigned long int &ra, unsigned long int &rb);
	static bool twoTargets(unsigned long int ir);
	static std::vector<unsigned long int> executeALU(unsigned long int ir, unsigned long int ra, unsigned long int rb, unsigned long int imm, bool &overflow);
};