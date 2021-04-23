package com.rstewart61.ml.project4;

import java.util.List;

import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;

public class ExperimentDomain {
	String type;
	SADomain domain;
	State initialState;
	int size;
	int numStates;
	List<State> reachableStates;
}
