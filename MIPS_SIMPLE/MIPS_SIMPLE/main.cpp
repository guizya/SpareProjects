#include <iostream>
#include "Pipeline.h"
#include <iostream>

using namespace std;

unsigned long int globalCycle = 0;

int main()
{
	int mode = 0;
	unsigned int num = 0;
	unsigned int startPC = 0;
	LatchReg::intialize();
	LatchReg::PC = startPC;

	unsigned int endCycle = 0xffffffff;
	unsigned int endPC = 0xffffffff;

	Pipeline pipe;
	
	try {
		Memory::initialize("D:\\projects\\MIPS_SIMPLE\\assembly.txt");

		cout << "Choose to select instruction mode or cycle mode: 0 - cycle, 1 - instruction" << std::endl;
		cin >> mode;

		while (true)
		{
			if (num == 0) {
				cout << "Choose the number of " << (mode == 0 ? "cycles" : "instructions") << " you want to run:" << std::endl;
				cin >> num;
			}

			startPC = pipe.getPCcount();

			pipe.WB(globalCycle);
			pipe.MEM(globalCycle);
			pipe.EX(globalCycle);
			pipe.ID(globalCycle);
			pipe.IF(globalCycle);

			globalCycle++;

			if (mode == 0) num--;	// cycle mode
			else {
				if (pipe.getPCcount() > startPC) num--;	// instruction mode
			}

			//cout << "Cycle " << globalCycle << " Pipe status: " << std::endl;
			//cout << pipe.getString(globalCycle) << std::endl;

			if (num == 0) {
				cout << "Cycle " << globalCycle << " Pipe status: " << std::endl;
				cout << pipe.getString(globalCycle) << std::endl;

				cout << "Continue ? 0 - stop, 1 - continue" << std::endl;
				int value = 0;
				cin >> value;
				if (value == 0) break;
			}
		}
	}
	catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}

	system("pause");
	return 0;
}