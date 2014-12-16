// AEP 4380 Final Project
// Gagik Hakobyan
//
// Conducting molecular dynamics simulations on octanes on a GPU.
//
// Run on a Core i7 and a GTX 580 GPU with Visual Studio 2013 and nvcc in Windows 8.1
//
// 12-15-2014
#include "supercell.h"	
#include "mainIterator.cuh"
#include <iostream>

using namespace std;


void main() {
	supercell superCell(8, "test2.mol2");
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