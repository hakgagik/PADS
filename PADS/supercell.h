#include <armadillo>
#include <iostream>

// Contains various constants having to do with alkanes
namespace mol {
	const double mC = 1.99442E-26;
	const double mH = 1.67372E-27;
	const double mCH3 = mC + 2.0 * mH;
	const double mCH4 = mC + 3.0 * mH;

	const double ang = 1E-10;
	const double lCH = 1.09;
	const double lCHang = 1.09*ang;
}

// A class for storing molecules. These will later be converted into arrays to be processed on the GPU.
class molecule {
public:
	molecule();
	molecule(int n, arma::mat &atoms);

	int nBeads;
	arma::mat beads;
	arma::rowvec centroid;
};


class supercell {
public:
	supercell();
	supercell(int nC, std::string infile);

	int nMols;
	molecule* mols;
};