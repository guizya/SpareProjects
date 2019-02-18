#include "LatchReg.h"
#include <sstream>

unsigned long int LatchReg::PC;
unsigned long int LatchReg::R[32];

void LatchReg::intialize()
{
	PC = 0;
	memset(R, 0, 32 * sizeof(unsigned long int));
}

LatchReg::LatchReg()
{
	reset();
}

void LatchReg::reset()
{
	IR = 0;
	NPC = 0;
	A = 0;
	B = 0;
	Imm = 0;
	ALUoutput = 0;
	ALUoutput_high = 0;
	cond = 0;
	LMD = 0;
	available = true;
}

void LatchReg::setIR(unsigned long int t)
{
	IR = t;
}

void LatchReg::setNPC(unsigned long int t)
{
	NPC = t;
}

void LatchReg::setA(unsigned long int t)
{
	A = t;
}

void LatchReg::setB(unsigned long int t)
{
	B = t;
}

void LatchReg::setImm(unsigned long int t)
{
	Imm = t;
}

void LatchReg::setALUoutput(unsigned long int t)
{
	ALUoutput = t;
}

void LatchReg::setALUoutput_high(unsigned long int t)
{
	ALUoutput_high = t;
}

void LatchReg::setcond(unsigned long int t)
{
	cond = t;
}

void LatchReg::setLMD(unsigned long int t)
{
	LMD = t;
}

void LatchReg::setAvailable(bool b)
{
	available = b;
}

unsigned long int LatchReg::getIR()
{
	return IR;
}

unsigned long int LatchReg::getNPC()
{
	return NPC;
}

unsigned long int LatchReg::getA()
{
	return A;
}

unsigned long int LatchReg::getB()
{
	return B;
}

unsigned long int LatchReg::getImm()
{
	return Imm;
}

unsigned long int LatchReg::getALUoutput()
{
	return ALUoutput;
}

unsigned long int  LatchReg::getALUoutput_high()
{
	return ALUoutput_high;
}

unsigned long int LatchReg::getcond()
{
	return cond;
}

unsigned long int LatchReg::getLMD()
{
	return LMD;
}

bool LatchReg::getAvailable() {
	return available;
}

std::string LatchReg::getString() 
{
	std::ostringstream out;
	out << std::hex;
	out << "IR: 0x" << IR << " ";
	out << "NPC: 0x" << NPC << " ";
	out << "A: 0x" << A << " ";
	out << "B: 0x" << B << " ";
	out << "Imm: 0x" << Imm << " ";
	out << "ALUoutput: 0x" << ALUoutput << " ";
	out << "cond: 0x" << cond << " ";
	out << "LMD: 0x" << LMD << " ";
	out << std::dec;
	return out.str();
}

std::string LatchReg::enumStr(Register reg)
{
	switch (reg) {
	case reg_zero:
		return "zero";
	case reg_at:
		return "at";
	case reg_v0:
		return "v0";
	case reg_v1:
		return "v1";
	case reg_a0:
		return "a0";
	case reg_a1:
		return "a1";
	case reg_a2:
		return "a2";
	case reg_a3:
		return "a3";
	case reg_t0:
		return "t0";
	case reg_t1:
		return "t1";
	case reg_t2:
		return "t2";
	case reg_t3:
		return "t3";
	case reg_t4:
		return "t4";
	case reg_t5:
		return "t5";
	case reg_t6:
		return "t6";
	case reg_t7:
		return "t7";
	case reg_s0:
		return "s0";
	case reg_s1:
		return "s1";
	case reg_s2:
		return "s2";
	case reg_s3:
		return "s3";
	case reg_s4:
		return "s4";
	case reg_s5:
		return "s5";
	case reg_s6:
		return "s6";
	case reg_s7:
		return "s7";
	case reg_t8:
		return "t8";
	case reg_t9:
		return "t9";
	case reg_k0:
		return "k0";
	case reg_k1:
		return "k1";
	case reg_gp:
		return "gp";
	case reg_sp:
		return "sp";
	case reg_fp:
		return "fp";
	case reg_ra:
		return "ra";
	}

	return "";
}

std::string LatchReg::getStaticString()
{
	std::ostringstream out;
	out << std::hex;
	out << "PC: 0x" << PC << " ";
	for (unsigned long int i = 0; i < 32; ++i)
		out << enumStr((Register)i) << ": 0x" << R[i] << " ";
	out << std::dec;
	return out.str();
}