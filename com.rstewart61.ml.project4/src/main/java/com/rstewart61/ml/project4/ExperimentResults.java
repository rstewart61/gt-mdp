package com.rstewart61.ml.project4;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class ExperimentResults {
	public int problemSize = -1;
	public int numStates = -1;
	public int iterations = -1;
	public int subIterations = -1;
	public int bellmanInvocations = -1;
	public int numSteps = -1;
	public double averageNumSteps = 0.0;
	public double averageReward = 0.0;
	public double discountFactor;
	public long cpuTime;
	public List<Double> viDeltaByIteration = new ArrayList<>();
	public List<Double> piDeltaByIteration = new ArrayList<>();
	
	@Override
	public String toString() {
		return "ExperimentResults [problemSize=" + problemSize + ", numStates=" + numStates + ", iterations="
				+ iterations + ", subIterations=" + subIterations + ", bellmanInvocations=" + bellmanInvocations
				+ ", numSteps=" + numSteps + ", averageNumSteps=" + averageNumSteps + ", averageReward=" + averageReward
				+ ", discountFactor=" + discountFactor + "]";
	}
	
	public void write(PrintWriter pw) {
		pw.format("%.4f,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f\n", discountFactor, problemSize,
				numStates, iterations, subIterations, bellmanInvocations, numSteps, cpuTime,
				averageNumSteps, averageReward);
	}
	
	public static void writeFile(String filename, List<ExperimentResults> results) throws IOException {
		FileWriter fw = new FileWriter(filename);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter pw = new PrintWriter(bw);
        pw.format("Discount Factor,Problem Size,Num States,Iterations," +
        		"Sub Iterations,Bellman Invocations,Num Steps,CPU Time,Average Num Steps,Average Reward\n");
        for (ExperimentResults result : results) {
        	result.write(pw);
        }
        pw.close();
	}
}
