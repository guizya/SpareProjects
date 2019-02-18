#include "Pipeline.h"
#include "Instruction.h"
#include <sstream>
#include <iostream>

void Pipeline::IF(unsigned long int cycle)
{
	// waiting for branch resolving
	if (signal.isStopFetching(cycle)) {
		IF_ID.reset();
		return;
	}

	if (!IF_ID.getAvailable()) {
		return;
	}

	bool branch = false;
	unsigned long int address = 0;
	signal.branchResolved(branch, address, cycle);
	if (branch) {
		LatchReg::PC = address;
	}

	unsigned long int ir = Memory::readInstructionMemory(LatchReg::PC);
	IF_ID.setIR(ir);
	IF_ID.setNPC(LatchReg::PC + 4);
	LatchReg::PC += 4;

	signal.countIF();
}

void Pipeline::ID(unsigned long int cycle)
{
	unsigned long int ir = IF_ID.getIR();

	if (Instruction::isNop(ir)) {
		ID_EX.reset();
		return;
	}

	if (Instruction::isBRA(ir)) {
		signal.stopFetching(cycle);
	}

	unsigned long int ra = 0, rb = 0;
	Instruction::getSourceRegisters(ir, ra, rb);
	
	if (signal.registerHazard(ra, rb, *this)) {
		ID_EX.reset();
		IF_ID.setAvailable(false);
		return;
	}
	else {
		IF_ID.setAvailable(true);
	}


	ID_EX.setA(LatchReg::R[ra]);
	ID_EX.setB(LatchReg::R[rb]);
	ID_EX.setNPC(IF_ID.getNPC());
	ID_EX.setIR(IF_ID.getIR());
	ID_EX.setImm(Instruction::getImm(ir));
	
	signal.countID();
}

void Pipeline::EX(unsigned long int cycle)
{
	unsigned long int ir = ID_EX.getIR();

	if (Instruction::isNop(ir)) {
		EX_MEM.reset();
		return;
	}

	EX_MEM.setIR(ir);

	// ALU instruction
	if (Instruction::isALU(ir)) {
		bool overflow = false;
		std::vector<unsigned long int> results = Instruction::executeALU(ir, ID_EX.getA(), ID_EX.getB(), ID_EX.getImm(), overflow);

		EX_MEM.setALUoutput(results[0]);
		if (results.size() == 2) {
			EX_MEM.setALUoutput_high(results[1]);
		}
		EX_MEM.setcond(overflow);	// reuse cond for overflow case
	}

	// Load Store instructions
	if (Instruction::isLoadStore(ir)) {
		EX_MEM.setALUoutput(ID_EX.getA() + ID_EX.getImm());
		EX_MEM.setB(ID_EX.getB());
	}

	// Branch instructions
	if (Instruction::isBRA(ir)) {
		EX_MEM.setALUoutput(ID_EX.getNPC() + (ID_EX.getImm() << 2));
		EX_MEM.setcond(ID_EX.getA() == ID_EX.getB()); /*NOTE: beq only currently, so this is sufficient*/

		signal.signalIF(EX_MEM.getcond(), EX_MEM.getALUoutput(), cycle);
	}
	EX_MEM.setNPC(ID_EX.getNPC());

	signal.countEX();
}

void Pipeline::MEM(unsigned long int cycle)
{
	unsigned long int ir = EX_MEM.getIR();

	if (Instruction::isNop(ir)) {
		MEM_WB.reset();
		return;
	}

	MEM_WB.setIR(ir);

	// ALU instruction
	if (Instruction::isALU(ir)) {
		MEM_WB.setALUoutput(EX_MEM.getALUoutput());
		MEM_WB.setALUoutput_high(EX_MEM.getALUoutput_high());
	}

	// Load store instruction
	if (Instruction::isLoadStore(ir)) {
		// Load
		if (Instruction::isLoad(ir)) {
			MEM_WB.setLMD(Memory::readDataMemory(EX_MEM.getALUoutput()));
		}

		// Store
		if (Instruction::isStore(ir)) {
			Memory::writeDataMemory(EX_MEM.getALUoutput(), EX_MEM.getB());	
		}
	}

	MEM_WB.setNPC(EX_MEM.getNPC());

	signal.countMEM();
}

void Pipeline::WB(unsigned long int cycle)
{
	unsigned long int ir = MEM_WB.getIR();
	if (!Instruction::isNop(ir)) {
		pcCount++;
	}

	unsigned long int rd = Instruction::getTargetRegister(ir);

	if (rd == 0) return;

	// ALU instruction
	if (Instruction::isALU(ir)) {
		if (MEM_WB.getcond()) return;	// in overflow case, do nothing

		LatchReg::R[rd] = MEM_WB.getALUoutput();
		if (Instruction::twoTargets(ir)) {
			LatchReg::R[rd + 1] = MEM_WB.getALUoutput_high();
		}
	}

	// Load instruction
	if (Instruction::isLoad(ir)) {
		LatchReg::R[rd] = MEM_WB.getLMD();
	}

	signal.countWB();
}

std::string Pipeline::getString(unsigned long int cycle)
{
	std::ostringstream out;
	out << "IF/ID: " << IF_ID.getString() << std::endl;
	out << "ID/EX: " << ID_EX.getString() << std::endl;
	out << "EX/MEM: " << EX_MEM.getString() << std::endl;
	out << "MEM/WB: " << MEM_WB.getString() << std::endl;
	out << LatchReg::getStaticString() << std::endl;

	float ifUti, idUti, exUti, memUti, wbUti;
	signal.utilization(cycle, ifUti, idUti, exUti, memUti, wbUti);

	out << "Utilization IF: " << ifUti << " ID: " << idUti << " EX: " << exUti << " MEM: " << memUti << " WB: " << wbUti << std::endl;

	return out.str();
}

unsigned long int Pipeline::getPCcount()
{
	return pcCount;
}