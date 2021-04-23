package com.rstewart61.ml.project4;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.DynamicProgramming;
import burlap.mdp.core.state.State;
import burlap.statehashing.HashableState;

abstract public class PlanningIteration extends DynamicProgramming implements Planner {
	protected int													totalValueIterations = 0;
	protected int													totalStatesConsidered = 0;
	protected int													reachableStates = 0;
	protected IterationCallback valueCallback = null;
	protected IterationCallback policyCallback = null;
	
	public interface IterationCallback {
		public void recordIteration(PlanningIteration planner, int iteration, double delta);
	}
	
	public void setValueIterationCallback(IterationCallback callback) {
		this.valueCallback = callback;
	}
	
	public void setPolicyIterationCallback(IterationCallback callback) {
		this.policyCallback = callback;
	}
		
	public int getTotalValueIterations() {
		return totalValueIterations;
	}
	
	public int getTotalStatesConsidered() {
		return totalStatesConsidered;
	}
	
	public int getNumReachableStates() {
		return reachableStates;
	}
	
	protected void triggerValueIterationCallback(int iteration, double delta) {
		if (valueCallback != null) {
			valueCallback.recordIteration(this, iteration, delta);
		}
	}
	
	protected void triggerPolicyIterationCallback(int iteration, double delta) {
		if (policyCallback != null) {
			policyCallback.recordIteration(this, iteration, delta);
		}
	}

	public Collection<State> getReachableStates() {
		List<State> result = new ArrayList<>();
		for (HashableState sh : this.valueFunction.keySet()) {
			result.add(sh.s());
		}
		return result;
	}
}
