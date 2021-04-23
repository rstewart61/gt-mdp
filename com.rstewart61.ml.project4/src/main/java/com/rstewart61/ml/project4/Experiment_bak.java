package com.rstewart61.ml.project4;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.PolicyRenderLayer;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionRenderLayer;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.MultiLayerRenderer;

public class Experiment_bak {
	final static String outputFolder = "/home/brandon/Dropbox/coursework/ml/project4/output/";
	final static int NUM_TRIALS = 50;
	final static int MAX_STEPS = 200000;
	final static int TRIAL_LENGTH = 5000;
	final static int MAX_ITERATIONS = 10000;
	final static double MAX_DELTA = 0.001;
	OOSADomain domain;
	State initialState;
	Collection<State> reachableStates;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	int width, height;
	double learningRate, discountFactor, qInit;

	public Experiment_bak(int size, double discountFactor, double learningRate, double qInit) {
		this.width = this.height = size;
		this.learningRate = learningRate;
		this.discountFactor = discountFactor;
		this.qInit = qInit;
		GridWorldDomain gwdg;
		gwdg = new GridWorldDomain(this.width, this.height);
		// gwdg.setMapToFourRooms();
		
		int split = size / 2;
		int opening1 = size / 3;
	    int opening2 = size * 2 / 3;
		
		gwdg.verticalWall(0, opening1 - 1, split);
		gwdg.verticalWall(opening1+1, opening2-1, split);
		gwdg.verticalWall(opening2+1, size-1, split);
		
		gwdg.horizontalWall(0, opening1 - 1, split);
		gwdg.horizontalWall(opening1+1, opening2-1, split);
		gwdg.horizontalWall(opening2+1, size-1, split);

	
		TerminalFunction tf;
		tf = new GridWorldTerminalFunction(this.width - 1, this.height - 1);
		gwdg.setTf(tf);
		StateConditionTest goalCondition;
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();

		initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(this.width - 1, this.height - 1, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
		
		// Use a ValueIteration just to get reachable states for now
		ValueIteration planner = new ValueIteration(domain, 0, hashingFactory, 0, 0);
		planner.performReachabilityFrom(initialState);
		this.reachableStates = planner.getReachableStates();
		

		// VisualActionObserver observer = new VisualActionObserver(domain,
		// GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		// observer.initGUI();
		// env.addObservers(observer);
	}
	
	public int getNumStates() {
		return this.width * this.height;
	}

	/*
	public void visualize(String outputpath) {
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}
	*/
	
	private double getAverageNumSteps(GreedyQPolicy policy) {
		int total = 0;
		for (int i=0; i<NUM_TRIALS; ++i) {
			int numSteps = getNumSteps(policy, initialState);
			total += numSteps;
		}
		return (double)total / (double) NUM_TRIALS;
	}
	
	private String stateString(State state) {
		int x = ((GridWorldState)state).agent.x;
		int y = ((GridWorldState)state).agent.y;
		return "{" + x + ", " + y + "}";
	}
	
	private int getNumSteps(GreedyQPolicy policy, State state) {
		env.resetEnvironment();
		env.setCurStateTo(state);
		int numSteps = 0;
		int maxSteps = getNumStates();
		Map<State, Integer> visited = new HashMap<>();
		
		while(!env.isInTerminalState() && numSteps < maxSteps) {
			if (!visited.containsKey(state)) {
				visited.put(state, 0);
			}
			if (visited.get(state) > 5) {
				// cycles means probably not converged
				return maxSteps;
			}
			visited.put(state, visited.get(state) + 1);
			// System.out.println(stateString(state));
			Action a = policy.action(state);
			EnvironmentOutcome eo = env.executeAction(a);
			state = eo.op;
			++numSteps;
		}

		return numSteps;
	}

	
	public ExperimentResults valueIterationExample(String outputPath) {
		ValueIteration planner = new ValueIteration(domain, discountFactor,
				hashingFactory, MAX_DELTA, MAX_ITERATIONS);
		GreedyQPolicy policy = planner.planFromState(initialState);

		ExperimentResults results = new ExperimentResults();
		results.discountFactor = discountFactor;
		results.problemSize = this.getNumStates();
		results.numStates = planner.getNumReachableStates();
		results.iterations = planner.getTotalValueIterations();
		results.bellmanInvocations = planner.getTotalStatesConsidered();
		results.numSteps = getNumSteps(policy, initialState);
		results.averageNumSteps = getAverageNumSteps(policy);
		results.averageReward = runPolicy(policy);
		
		// PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath +
		// "vi");

		// simpleValueFunctionVis((ValueFunction)planner, p);
		// manualValueFunctionVis((ValueFunction) planner, p, outputPath + "value_" + this.width + "x" + this.height);

		return results; 
	}

	public ExperimentResults policyIterationExample(String outputPath) {
		PolicyIteration planner = new PolicyIteration(domain, discountFactor,
				hashingFactory, MAX_DELTA, MAX_ITERATIONS, MAX_ITERATIONS);
		GreedyQPolicy policy = planner.planFromState(initialState);


		ExperimentResults results = new ExperimentResults();
		results.discountFactor = discountFactor;
		results.problemSize = this.getNumStates();
		results.numStates = planner.getNumReachableStates();
		results.iterations = planner.getTotalPolicyIterations();
		results.subIterations = planner.getTotalValueIterations();
		results.bellmanInvocations = planner.getTotalStatesConsidered();
		results.averageNumSteps = getAverageNumSteps(policy);
		results.averageReward = runPolicy(policy);
		results.numSteps = getNumSteps(policy, initialState);
		
		// simpleValueFunctionVis((ValueFunction)planner, p);
		// manualValueFunctionVis((ValueFunction) planner, p, outputPath + "policy_" + this.width + "x" + this.height);
		return results;
	}
	
	public double runPolicy(GreedyQPolicy policy) {
		env.resetEnvironment();
		env.setCurStateTo(initialState);

		State state = initialState;
		double reward = 0.0;
		int numSteps = 0;
		while (!env.isInTerminalState() && numSteps < 100000) {
			Action a = policy.action(state);
			EnvironmentOutcome outcome = env.executeAction(a);
			state = outcome.op;
			reward += outcome.r;
			++numSteps;
		}
		return reward;
	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p) {
		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, 11, 11,
				valueFunction, p);
		gui.initGUI();
	}

	private class SimpleJFrame extends JFrame {
		MultiLayerRenderer visualizer;

		public SimpleJFrame(MultiLayerRenderer visualizer) {
			this.visualizer = visualizer;
		}

		public void initGUI() {
			this.visualizer.setPreferredSize(new Dimension(800, 800));
			this.visualizer.setBGColor(Color.GRAY);

			this.getContentPane().add(visualizer, BorderLayout.CENTER);
			pack();
			setVisible(true);

			this.visualizer.repaint();
		}
	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p, String outputPath) {

		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

		// define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		// define a 2D painter of state values,
		// specifying which attributes correspond to the x and y coordinates of the
		// canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), new VariableDomain(0, 11), 1, 1);

		// create our ValueFunctionVisualizer that paints for all states
		// using the ValueFunction source and the state value painter we defined
		// ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates,
		// svp, valueFunction);

		// define a policy painter that uses arrow glyphs for each of the grid world
		// actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), new VariableDomain(0, 11), 1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);

		MultiLayerRenderer visualizer = new MultiLayerRenderer();
		ValueFunctionRenderLayer vfLayer = new ValueFunctionRenderLayer(allStates, svp, valueFunction);
		PolicyRenderLayer pLayer = new PolicyRenderLayer(allStates, spp, p);

		visualizer.addRenderLayer(vfLayer);
		visualizer.addRenderLayer(pLayer);
		// visualizer.setBGColor(Color.GRAY);

		SimpleJFrame frame = new SimpleJFrame(visualizer);
		frame.setTitle(outputPath);
		frame.initGUI();

		BufferedImage image = new BufferedImage(visualizer.getWidth(), visualizer.getHeight(),
				BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics2D = image.createGraphics();
		visualizer.repaint();
		visualizer.paint(graphics2D);
		SwingUtilities.invokeLater(() -> {
			try {
				// Sleep added to avoid graphics corruption in saved images.
				Thread.sleep(500); // TODO: Do something better here
				ImageIO.write(image, "png", new File(outputPath + ".png"));
			} catch (Exception e) {
				System.err.println("Could not write to " + outputPath);
				e.printStackTrace();
			} finally {
				SwingUtilities.invokeLater(() -> {
					frame.dispose();
				});
			}
		});
	}

	public void experimentAndPlotter(String dirname) {
		// different reward function for more structured performance plots
		// ((FactoredModel) domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
			public String getAgentName() {
				return "Q-Learning";
			}

			public LearningAgent generateAgent() {
				return new QLearning(domain, discountFactor, hashingFactory, qInit, learningRate);
			}
		};

		// LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, NUM_TRIALS, TRIAL_LENGTH, qLearningFactory);
		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, NUM_TRIALS, MAX_STEPS, qLearningFactory);
		exp.toggleTrialLengthInterpretation(false);

		boolean doPlots = false;
		exp.toggleVisualPlots(doPlots);
		if (doPlots) {
			exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOST_RECENT_AND_AVERAGE,
					PerformanceMetric.AVERAGE_EPISODE_REWARD, PerformanceMetric.STEPS_PER_EPISODE);
		}

		exp.startExperiment();
		String fileName = String.format("%d,%.4f,%.2f,%.2f", this.getNumStates(), this.discountFactor,
				this.learningRate, this.qInit);
		String dirPath = outputFolder + "qlearning/" + dirname;
		File directory = new File(dirPath);
	    if (!directory.exists()) {
	    	directory.mkdir();
	    }
		
		exp.writeEpisodeDataToCSV(dirPath + "/" + fileName);
		System.out.println("Completed Q-Learning for " + fileName);
	}
	
	public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
		List<ExperimentResults> allValueResults = new ArrayList<>();
		List<ExperimentResults> allPolicyResults = new ArrayList<>();
		ExecutorService executor = Executors.newFixedThreadPool(32);
		List<Future<Void>> futures = new ArrayList<>();
		DPrint.toggleUniversal(false);
		
		// Vary size, use default choices for discount factor, learning rate, and q-init
		double discountFactor = 0.99;
		double learningRate = 0.1;
		double qInit = 0.3;
		for (int size = 10; size <= 50; size+=10) {
			System.out.println("Size: " + size);
			Experiment_bak example = new Experiment_bak(size, discountFactor, learningRate, qInit);

			ExperimentResults vResults = example.valueIterationExample(outputFolder);
			allValueResults.add(vResults);
			System.out.println("VALUE: " + vResults);
			ExperimentResults pResults = example.policyIterationExample(outputFolder);
			allPolicyResults.add(pResults);
			System.out.println("POLICY: " + pResults);

			Future<Void> future = executor.submit(() -> {
				example.experimentAndPlotter("Num States");
				return null;
			});
			futures.add(future);
			
			// example.visualize(outputPath);
		}
		
		ExperimentResults.writeFile(outputFolder + "value_by_state.csv", allValueResults);
		ExperimentResults.writeFile(outputFolder + "policy_by_state.csv", allPolicyResults);
		allValueResults.clear();
		allPolicyResults.clear();

		// Default size, vary discount factor
		int size=30;
		for (int i=1; i<=40; ++i) {
			discountFactor = 1.0 - (double)i/200.0;
			System.out.println("Discount factor: " + discountFactor);
			Experiment_bak example = new Experiment_bak(size, discountFactor, learningRate, qInit);
			ExperimentResults vResults = example.valueIterationExample(outputFolder);
			allValueResults.add(vResults);
			System.out.println("VALUE: " + vResults);
			ExperimentResults pResults = example.policyIterationExample(outputFolder);
			allPolicyResults.add(pResults);
			System.out.println("POLICY: " + pResults);

			Future<Void> future = executor.submit(() -> {
				example.experimentAndPlotter("Discount Factor");
				return null;
			});
			futures.add(future);
		}
		
		ExperimentResults.writeFile(outputFolder + "value_by_discount_factor.csv", allValueResults);
		ExperimentResults.writeFile(outputFolder + "policy_by_discount_factor.csv", allPolicyResults);

		// Default size, discount factor. Vary learning rate.
		discountFactor = 0.99;
		for (int i=1; i<=20; ++i) {
			learningRate = 1.0 - (double)i/20.0;
			System.out.println("Learning rate: " + learningRate);
			Experiment_bak example = new Experiment_bak(size, discountFactor, learningRate, qInit);

			Future<Void> future = executor.submit(() -> {
				example.experimentAndPlotter("Learning Rate");
				return null;
			});
			futures.add(future);
		}

		// Default size, discount factor, learning rate. Vary qInit.
		discountFactor = 0.99;
		learningRate = 0.1;
		for (int i=1; i<=20; ++i) {
			qInit = 1.0 - (double)i/20.0;
			System.out.println("Q-Init: " + qInit);
			Experiment_bak example = new Experiment_bak(size, discountFactor, learningRate, qInit);

			Future<Void> future = executor.submit(() -> {
				example.experimentAndPlotter("QInit");
				return null;
			});
			futures.add(future);
		}

		
		for (Future<Void> future : futures) {
			future.get();
		}
		
		executor.shutdown();
	}

}