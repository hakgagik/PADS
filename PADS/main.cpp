#include "supercell.h"	
#include <iomanip>

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

	cout << superCell.nMols * superCell.mols[0].nBeads * 3 * sizeof(double) << endl;
}