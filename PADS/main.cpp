#include "supercell.h"	
#include <iomanip>
#include "mainIterator.cuh"


using namespace std;

void main() {
	supercell superCell(8, "test.mol2");
	int nMols = superCell.nMols;
	int nBeads = superCell.mols[0].nBeads;
	double *x, *y, *z;
	x = new double[nMols * nBeads];
	y = new double[nMols * nBeads];
	z = new double[nMols * nBeads];

	superCell.toArray(x, y, z);

	cuMainLoop(x, y, z, nMols, nBeads);

	//ofstream testOut("testOut.dat");

	//for (int i = 0; i < superCell.nMols; i++) {
	//	for (int j = 0; j < superCell.mols[i].nBeads; j++){
	//		testOut << setw(15) << superCell.mols[i].beads(j, 0)
	//			<< setw(15) << superCell.mols[i].beads(j, 1)
	//			<< setw(15) << superCell.mols[i].beads(j, 2)
	//			<< endl;
	//	}
	//}

	//cout << cuMain() << endl;
}