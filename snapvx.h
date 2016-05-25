#include "Snap.h"
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <boost/python.hpp>

namespace SnapVX
{

struct CVXPYutil
{
	int varID;
	std::string varName;
	int variable;
	int offset;
};
		
class TGraphVX : public TUNGraph
{
	protected:
		//data structures	
		std::map<int,int> node_objectives;
		std::map<int,vector<int> > node_constraints;
		std::map<pair<int,int>,int > edge_objectives;
		std::map<pair<int,int>,vector<int> > edge_constraints;
		
		//ADMM specific data structures
		std::map<int,struct CVXPYutil> node_variables;
		std::map<int,vector<double> > node_values;

		//solver functions
		virtual void SolveADMM();
		virtual bool CheckConvergence();
		
	public:
		TGraphVX(TUNGraph *Graph = NULL);	//constructor to initialise the snapvx graph
		virtual void Solve() = 0;
		virtual double GetTotalProblemValue();
};
		
		
	

