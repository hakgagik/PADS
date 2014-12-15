#include "supercell.h"

using namespace std;
using namespace arma;

molecule::molecule(){
}

// Read in positions of all the hydrogens and carbons and consolidate them into beads.
// Parameters:
// n: the total number of CARBON atoms
// atoms: the input x,y,z positions of all atoms in the actual alkane
// NOTE: For now, the position of each bead is its COM.
// Structure of atoms should be {HCHH CHH CHH CHH ... CHHH}
molecule::molecule(int n, mat &atoms){
	nBeads = n;
	beads.resize(n, 3);
	centroid = rowvec(3);
	beads.fill(fill::zeros);
	centroid.fill(fill::zeros);

	double mTotal = 2.0 * mol::mH;

	for (int i = 0; i < n; i++){
		if (i == 0) {
			beads.row(i) += atoms.row(0) * mol::mH;
		}

		int index = 1 + i * 3;
		beads.row(i) += atoms.row(index) * mol::mC;
		beads.row(i) += atoms.row(index + 1) * mol::mH;
		beads.row(i) += atoms.row(index + 2) * mol::mH;

		if (i == n - 1) {
			beads.row(i) += atoms.row(index + 3) * mol::mH;
		}
		centroid += beads.row(i);
		mTotal += mol::mCH3;

		if (i == 0 || i == n - 1) beads.row(i) /= mol::mCH4;
		else beads.row(i) /= mol::mCH3;
	}
	centroid /= mTotal;
	
	// Sort beads by z-coordinate. Yes. I've gotten to that point. If I hade more time, this would be a lot more intracate :P
	// Bubblesort time!
	// Aint nobody got time for quicksort.
	rowvec dummy;
	for (int i = 0; i < nBeads - 1; i++){
		for (int j = i; j < nBeads - 1; j++) {
			if (beads(j, 2) > beads(j + 1, 2)) {
				dummy = beads.row(j);
				beads.row(j) = beads.row(j + 1);
				beads.row(j + 1) = dummy;
			}
		}
	}
}

supercell::supercell(){
}

// Load a mol2 file and sort it into molecules for futher processing.
// Params:
// nC: number of carbon atoms in each alkane, assumes one alkane per cell.
// infile: input mol2 file
supercell::supercell(int nC, string infile) {
	ifstream fp(infile);
	int atomsPerMol = 2 + 3 * nC;

	char * isCarbon = new char[nC];

	string dummy;
	string elt;

	if (fp.bad()){
		cout << "Can't upen input file: " << infile << endl;
		exit(EXIT_FAILURE);
	}

	// Skip header line, except for one line which tells us the total number of atoms;
	for (int i = 0; i < 3; i++){
		getline(fp, dummy);
	}

	int nAtoms;
	fp >> nAtoms;
	nMols = nAtoms / atomsPerMol;
	getline(fp, dummy);

	mat atoms(nAtoms, 3);
	int* cIndex = new int[nC];
	int cCount = 0;

	while (true) {
		getline(fp, dummy);
		if (dummy == "@<TRIPOS>ATOM") break;
	}

	// Read through and copy all the atoms.
	for (int i = 0; i < nAtoms; i++){
		fp >> dummy;

		fp >> elt;
		fp >> atoms(i, 0); fp >> atoms(i, 1); fp >> atoms(i, 2);

		getline(fp, dummy);

		if (i < atomsPerMol){
			if (elt[0] == 'C') {
				isCarbon[i] = 1;
				cIndex[cCount] = i;
				cCount++;
			}
			else isCarbon[i] = 0;
		}
	}

	// Grab the first molecule to figure out the structure of the alkane
	mat testMol = atoms.submat(0, 0, atomsPerMol, 2);
	rowvec diff(3);
	imat CHgroups(nC, 4); CHgroups.fill(-1);
	int* nH = new int[nC];
	for (int i = 0; i < nC; i++){
		int carbon = cIndex[i];
		int hCount = 1;
		CHgroups(i, 0) = carbon;
		for (int j = 0; j < atomsPerMol; j++){
			diff = testMol.row(j) - testMol.row(carbon);
			if (norm(diff) < mol::lCH * 1.1 && carbon != j) {
				CHgroups(i, hCount) = j;
				hCount++;
			}
		}
	}

	int* order = new int[atomsPerMol];
	bool firstSet = false;
	int index = 4;

	for (int i = 0; i < nC; i++){
		if (CHgroups(i, 3) != -1) {
			if (!firstSet) {
				order[0] = CHgroups(i, 3);
				order[1] = CHgroups(i, 0);
				order[2] = CHgroups(i, 1);
				order[3] = CHgroups(i, 2);
				firstSet = true;
			}
			else {
				order[atomsPerMol - 1] = CHgroups(i, 3);
				order[atomsPerMol - 4] = CHgroups(i, 0);
				order[atomsPerMol - 3] = CHgroups(i, 1);
				order[atomsPerMol - 2] = CHgroups(i, 2);
			}
		}
		else {
			order[index] = CHgroups(i, 0);
			order[index + 1] = CHgroups(i, 1);
			order[index + 2] = CHgroups(i, 2);
			index += 3;
		}
	}

	mat thisMol(atomsPerMol, 3);
	mols = new molecule[nMols];

	for (int i = 0; i < nMols; i++){
		index = atomsPerMol * i;
		for (int j = 0; j < atomsPerMol; j++){
			thisMol.row(j) = atoms.row(order[j] + index);
		}
		mols[i] = molecule(nC, thisMol);
	}
}

void supercell::toArray(double *x, double *y, double *z){
	int nBeads = mols[0].nBeads;
	for (int i = 0; i < nMols; i++){
		for (int j = 0; j < nBeads; j++){
			x[i * nBeads + j] = mols[i].beads(j, 0) / 10; // Converting angstroms to nm
			y[i * nBeads + j] = mols[i].beads(j, 1) / 10;
			z[i * nBeads + j] = mols[i].beads(j, 2) / 10;
		}
	}
}