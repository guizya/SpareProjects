#include "Instruction.h"
#include <iostream>

unsigned long int Instruction::special(unsigned long int ir)
{
	return (ir >> 26) & 0x3f;
}

unsigned long int Instruction::rs(unsigned long int ir)
{
	return (ir >> 21) & 0x1f;
}

unsigned long int Instruction::rt(unsigned long int ir)
{
	return (ir >> 16) & 0x1f;
}

unsigned long int Instruction::rd(unsigned long int ir)
{
	return (ir >> 11) & 0x1f;
}

unsigned long int Instruction::opcode(unsigned long int ir)
{
	return ir & 0x3f;
}

unsigned long int Instruction::immediate(unsigned long int ir)
{
	return ir & 0xffff;
}

unsigned long int Instruction::sa(unsigned long int ir)
{
	return (ir >> 6) & 0x3f;
}

unsigned long int Instruction::signExtend(unsigned long int imm)
{
	unsigned long int signBit = (imm >> 15) & 1;
	if (signBit == 1) return 0xffff0000 | imm;
	return imm;
}

unsigned long int Instruction::zeroExtend(unsigned long int imm)
{
	return imm;
}

bool Instruction::check_overflow(unsigned long long int value)
{
	unsigned long int v = (unsigned long int)(value >> 32);
	if (v == 0 || v == 0xffffffff) return false;
	return true;
}

bool Instruction::isALU(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	unsigned long int opcodeBits = opcode(ir);

	if (specialBits == 0) {
		switch (opcodeBits) {
		case ADD:	// add
		case SUB:	// sub
		case MUL:	// mult
		case AND:	// and
		case OR:	// or
		case SLL:		// sll
		case SRL:		// srl
			return true;
		}
	}

	switch (specialBits) {
	case ADDI:		// addi
	case LUI:	// lui
	case ANDI:	// andi
	case ORI:	// ori
	case SLTI:	// slti
	case SLTIU:	// sltiu
		return true;
	}

	return false;
}

bool Instruction::isBRA(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	if (specialBits == BEQ) return true;	// beq
	return false;
}

bool Instruction::isLoadStore(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	if (specialBits == LW || specialBits == SW) return true; // lw, sw
	return false;
}

bool Instruction::isLoad(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	if (specialBits == LW) return true; // lw
	return false;
}

bool Instruction::isStore(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	if (specialBits == SW) return true; // sw
	return false;
}

bool Instruction::isNop(unsigned long int ir)
{
	return ir == 0;
}

bool Instruction::hasImm(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);

	switch (specialBits) {
	case ADDI:		// addi
	case LUI:	// lui
	case ANDI:	// andi
	case ORI:	// ori
	case SLTI:	// slti
	case SLTIU:	// sltiu
		return true;
	case BEQ:		// beq
		return true;
	case LW:	// lw
	case SW:	// sw
		return true;
	}

	return false;
}

unsigned long int Instruction::getImm(unsigned long int ir)
{
	unsigned long int specialBits = special(ir);
	unsigned long int imm = immediate(ir);

	unsigned long int immSign = signExtend(imm);
	unsigned long int immZero = zeroExtend(imm);

	switch (specialBits) {
	case ADDI:
	case SLTI:
	case SLTIU:
	case BEQ:
	case LW:
	case SW:
		return immSign;

	case ANDI:
	case ORI:
		return immZero;

	case LUI:
		return (imm << 16);
	}

	return 0;
}

unsigned long int Instruction::getTargetRegister(unsigned long int ir)
{
	unsigned long int rtBits = rt(ir);
	unsigned long int rdBits = rd(ir);

	unsigned long int specialBits = special(ir);
	unsigned long int opcodeBits = opcode(ir);

	if (specialBits == 0) {
		switch (opcodeBits) {
		case ADD:
		case SUB:
		case AND:
		case OR:
		case SLL:
		case SRL:
		case MUL:
			return rdBits;
		}
	}

	switch (specialBits) {
	case ADDI:
	case LW:
	case LUI:
	case ANDI:
	case ORI:
	case SLTI:
	case SLTIU:
		return rtBits;
	}

	return 0;
}

void Instruction::getSourceRegisters(unsigned long int ir, unsigned long int &ra, unsigned long int &rb)
{
	unsigned long int rsBits = rs(ir);
	unsigned long int rtBits = rt(ir);
	unsigned long int saBits = sa(ir);

	unsigned long int specialBits = special(ir);
	unsigned long int opcodeBits = opcode(ir);

	if (specialBits == 0) {
		switch (opcodeBits) {
		case ADD:
		case SUB:
		case AND:
		case OR:
		case MUL:
			ra = rsBits;
			rb = rtBits;
			return;

		case SLL:
		case SRL:
			ra = rtBits;
			rb = saBits;
			return;
		}
	}

	switch (specialBits) {
	case ADDI:
	case LW:
	case ANDI:
	case ORI:
	case SLTI:
	case SLTIU:
		ra = rsBits;
		return;
	
	case LUI:
		ra = 0;
		rb = 0;
		return;

	case SW:
	case BEQ:
		ra = rsBits;
		rb = rtBits;
		return;
	}

}

bool Instruction::twoTargets(unsigned long int ir) {
	unsigned long int specialBits = special(ir);
	unsigned long int opcodeBits = opcode(ir);

	if (specialBits == 0 && opcodeBits == MUL) return true;
	return false;
}

std::vector<unsigned long int> Instruction::executeALU(unsigned long int ir, unsigned long int ra, unsigned long int rb, unsigned long int imm, bool &overflow)
{
	std::vector<unsigned long int> result;

	unsigned long int specialBits = special(ir);
	unsigned long int opcodeBits = opcode(ir);

	long int operatorA = (long int)ra;
	long int operatorB = (long int)rb;
	long int opeartorI = (long int)imm;

	long long int value = 0;

	if (specialBits == 0 && opcodeBits == ADD) {
		value = operatorA + operatorB;
		overflow = check_overflow(value);
		result.push_back(value);
		return result;
	}

	if (specialBits == 0 && opcodeBits == SUB) {
		value = operatorA - operatorB;
		overflow = check_overflow(value);
		result.push_back(value);
		return result;
	}

	if (specialBits == ADDI) {
		value = operatorA + opeartorI;
		overflow = check_overflow(value);
		result.push_back(value);
		return result;
	}

	if (specialBits == 0 && opcodeBits == MUL) {
		value = (long long int) operatorA * (long long int) operatorB;
		result.push_back((value & 0xffffffff));
		result.push_back((value >> 32) & 0xffffffff);
		return result;
	}

	if (specialBits == LUI) {
		result.push_back(imm);
		return result;
	}

	if (specialBits == 0 && opcodeBits == AND) {
		result.push_back(ra & rb);
		return result;
	}

	if (specialBits == ANDI) {
		result.push_back(ra & imm);
		return result;
	}

	if (specialBits == 0 && opcodeBits == OR) {
		result.push_back(ra | rb);
		return result;
	}

	if (specialBits == ORI) {
		result.push_back(ra | imm);
		return result;
	}

	if (specialBits == 0 && opcodeBits == SLL) {
		value = ra << rb;
		result.push_back(value);
		return result;
	}

	if (specialBits == 0 && opcodeBits == SRL) {
		value = ra >> rb;
		result.push_back(value);
		return result;
	}

	if (specialBits == SLTI) {
		value = (operatorA < opeartorI);
		result.push_back(value);
		return result;
	}

	if (specialBits == SLTIU) {
		value = (ra < imm);
		result.push_back(value);
		return result;
	}
}