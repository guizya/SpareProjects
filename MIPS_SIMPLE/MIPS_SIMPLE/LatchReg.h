#pragma once
#include <string>

enum Register {
	reg_zero = 0,
	reg_at,
	reg_v0,
	reg_v1,
	reg_a0,
	reg_a1,
	reg_a2,
	reg_a3,
	reg_t0,
	reg_t1,
	reg_t2,
	reg_t3,
	reg_t4,
	reg_t5,
	reg_t6,
	reg_t7,
	reg_s0,
	reg_s1,
	reg_s2,
	reg_s3,
	reg_s4,
	reg_s5,
	reg_s6,
	reg_s7,
	reg_t8,
	reg_t9,
	reg_k0,
	reg_k1,
	reg_gp,
	reg_sp,
	reg_fp,
	reg_ra,
};

class LatchReg 
{
private:
	unsigned long int IR;
	unsigned long int NPC;
	unsigned long int A;
	unsigned long int B;
	unsigned long int Imm;
	unsigned long int ALUoutput;
	unsigned long int ALUoutput_high;
	unsigned long int cond;
	unsigned long int LMD;
	bool available;

public:
	LatchReg();
	static unsigned long int R[32];
	static unsigned long int PC;

	void setIR(unsigned long int t);
	void setNPC(unsigned long int t);
	void setA(unsigned long int t);
	void setB(unsigned long int t);
	void setImm(unsigned long int t);
	void setALUoutput(unsigned long int t);
	void setcond(unsigned long int t);
	void setLMD(unsigned long int t);
	void setALUoutput_high(unsigned long int t);
	void setAvailable(bool b);

	void reset();

	unsigned long int getIR();
	unsigned long int getNPC();
	unsigned long int getA();
	unsigned long int getB();
	unsigned long int getImm();
	unsigned long int getALUoutput();
	unsigned long int getcond();
	unsigned long int getLMD();
	unsigned long int getALUoutput_high();
	bool getAvailable();

	std::string getString();
	static std::string enumStr(Register reg);
	static std::string getStaticString();
	static void intialize();
};