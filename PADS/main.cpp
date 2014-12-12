#include "supercell.h"	
#include <iomanip>
#include "cudaTest.cuh"

using namespace std;

void main() {
	supercell superCell(8, "test.mol2");
	//ofstream testOut("testOut.dat");

	//for (int i = 0; i < superCell.nMols; i++) {
	//	for (int j = 0; j < superCell.mols[i].nBeads; j++){
	//		testOut << setw(15) << superCell.mols[i].beads(j, 0)
	//			<< setw(15) << superCell.mols[i].beads(j, 1)
	//			<< setw(15) << superCell.mols[i].beads(j, 2)
	//			<< endl;
	//	}
	//}

	cout << cuMain() << endl;
}