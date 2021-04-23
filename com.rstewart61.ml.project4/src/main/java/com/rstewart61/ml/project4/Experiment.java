package com.rstewart61.ml.project4;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import burlap.behavior.learningrate.ExponentialDecayLR;
import burlap.behavior.learningrate.LearningRate;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
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
import burlap.behavior.valuefunction.ValueFunction;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.MultiLayerRenderer;

public class Experiment {
	final static int NUM_TRIALS = 50;
	final static int MAX_STEPS = 200000;
	// final static int MAX_STEPS = 20000;
	// final static int TRIAL_LENGTH = 5000; // episodes per trial
	final static int MAX_ITERATIONS = 10000;
	final static int MAX_EPISODE_LENGTH = 3000;
	final static double MAX_VI_DELTA = 0.00001;
	final static double MAX_PI_DELTA = 0.00001;
	final static double MAX_QL_DELTA = 0.0001;
	SADomain domain;
	State initialState;
	Collection<State> reachableStates;
	int numReachableStates = 0;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	ExperimentDomain expDomain;
	double learningRate, learningRateDecay, discountFactor, qInit;
	int trialLength;
	String type;
	int size;
	ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
	GreedyQPolicy piPolicyViz, viPolicyViz, qlPolicyViz;

	public Experiment(ExperimentDomain expDomain, int trialLength, double discountFactor,
			double learningRate, double learningRateDecay,
			double qInit) {
		this.expDomain = expDomain;
		this.domain = expDomain.domain;
		this.initialState = expDomain.initialState;
		this.type = expDomain.type;
		this.size = expDomain.size;
		this.trialLength = trialLength;

		this.learningRate = learningRate;
		this.learningRateDecay = learningRateDecay;
		this.discountFactor = discountFactor;
		this.qInit = qInit;

		hashingFactory = new HashableStateFactory() {
			HashableStateFactory delegate = new SimpleHashableStateFactory();

			@Override
			public HashableState hashState(State s) {
				if (s instanceof TrebleCrossState) {
					return new HashableState() {
						State _s = s;
						@Override
						public State s() {
							return _s;
						}
						
						public int hashCode() {
							return _s.hashCode();
						}
						
						public boolean equals(Object other) {
							return _s.hashCode() == other.hashCode();
						}
					};
				}
				return delegate.hashState(s);
			}
		};

		env = new SimulatedEnvironment(domain, initialState);

		// Use a ValueIteration just to get reachable states for now
		ValueIteration planner = new ValueIteration(domain, 0, hashingFactory, 0, 0);
		planner.performReachabilityFrom(initialState);
		this.reachableStates = planner.getReachableStates();
		this.numReachableStates = reachableStates.size();

		// VisualActionObserver observer = new VisualActionObserver(domain,
		// GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		// observer.initGUI();
		// env.addObservers(observer);
	}

	public int getNumStates() {
		return numReachableStates;
	}

	/*
	 * public void visualize(String outputpath) { Visualizer v =
	 * GridWorldVisualizer.getVisualizer(gwdg.getMap()); new
	 * EpisodeSequenceVisualizer(v, domain, outputpath); }
	 */

	private double getAverageNumSteps(GreedyQPolicy policy) {
		long start = System.currentTimeMillis();
		int total = 0;
		for (int i = 0; i < NUM_TRIALS; ++i) {
			int numSteps = getNumSteps(policy, initialState);
			total += numSteps;
		}
		System.out.printf("getAverageNumSteps took %d ms\n", System.currentTimeMillis() - start);
		
		return (double) total / (double) NUM_TRIALS;
	}

	private String stateString(State state) {
		int x = ((GridWorldState) state).agent.x;
		int y = ((GridWorldState) state).agent.y;
		return "{" + x + ", " + y + "}";
	}

	private int getNumSteps(GreedyQPolicy policy, State state) {
		env.resetEnvironment();
		env.setCurStateTo(state);
		int numSteps = 0;
		int maxSteps = getNumStates();
		Map<State, Integer> visited = new HashMap<>();

		while (!env.isInTerminalState() && numSteps < maxSteps) {
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

	private static void mkdir(String path) {
		File directory = new File(path);
		if (!directory.exists()) {
			System.out.println("Creating directory: " + path);
			directory.mkdir();
		}
	}

	public ExperimentResults valueIterationExample(String outputPath) {
		ValueIteration planner = new ValueIteration(domain, discountFactor, hashingFactory, MAX_VI_DELTA, MAX_ITERATIONS);
		ExperimentResults results = new ExperimentResults();
		planner.setValueIterationCallback((p, iteration, delta) -> {
			results.viDeltaByIteration.add(delta);
		});
		long startCPUTime = threadBean.getCurrentThreadCpuTime();
		GreedyQPolicy policy = planner.planFromState(initialState);
		results.cpuTime = threadBean.getCurrentThreadCpuTime() - startCPUTime;

		results.discountFactor = discountFactor;
		results.problemSize = size;
		results.numStates = this.numReachableStates; // planner.getNumReachableStates();
		results.iterations = planner.getTotalValueIterations();
		results.bellmanInvocations = planner.getTotalStatesConsidered();
		// results.numSteps = getNumSteps(policy, initialState);
		results.averageNumSteps = getAverageNumSteps(policy);
		results.averageReward = runPolicy(policy);
		
		System.gc();

		// PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath +
		// "vi");

		// simpleValueFunctionVis((ValueFunction)planner, p);
		// manualValueFunctionVis((ValueFunction) planner, p, outputPath + "value_" +
		// this.width + "x" + this.height);

		return results;
	}

	public ExperimentResults policyIterationExample(String outputPath) {
		PolicyIteration planner = new PolicyIteration(domain, discountFactor, hashingFactory, MAX_PI_DELTA, MAX_VI_DELTA, MAX_ITERATIONS,
				MAX_ITERATIONS);
		ExperimentResults results = new ExperimentResults();
		/*
		planner.setValueIterationCallback((p, iteration, delta) -> {
			results.viDeltaByIteration.add(delta);
		});
		*/
		planner.setPolicyIterationCallback((p, iteration, delta) -> {
			results.piDeltaByIteration.add(delta);
		});
		long startCPUTime = threadBean.getCurrentThreadCpuTime();
		GreedyQPolicy policy = planner.planFromState(initialState);
		results.cpuTime = threadBean.getCurrentThreadCpuTime() - startCPUTime;

		results.discountFactor = discountFactor;
		results.problemSize = size;
		results.numStates = this.numReachableStates; // planner.getNumReachableStates();
		results.iterations = planner.getTotalPolicyIterations();
		results.subIterations = planner.getTotalValueIterations();
		results.bellmanInvocations = planner.getTotalStatesConsidered();
		results.averageNumSteps = getAverageNumSteps(policy);
		results.averageReward = runPolicy(policy);
		
		System.gc();
		// results.numSteps = getNumSteps(policy, initialState);

		// simpleValueFunctionVis((ValueFunction)planner, p);
		// manualValueFunctionVis((ValueFunction) planner, p, outputPath + "policy_" +
		// this.width + "x" + this.height);
		return results;
	}

	public double runPolicy(GreedyQPolicy policy) {
		long start = System.currentTimeMillis();
		env.resetEnvironment();
		env.setCurStateTo(initialState);

		State state = initialState;
		double reward = 0.0;
		int episodes = 1;
		int numSteps = 0;
		while (numSteps < 30000) {
			Action a = policy.action(state);
			EnvironmentOutcome outcome = env.executeAction(a);
			state = outcome.op;
			reward += outcome.r;
			++numSteps;
			if (env.isInTerminalState()) {
				++episodes;
				env.resetEnvironment();
				state = initialState;
				env.setCurStateTo(initialState);
			}
		}
		System.out.printf("runPolicy took %d ms\n", System.currentTimeMillis() - start);
		return reward / episodes;
	}
	
	private HashableState h(State s) {
		return hashingFactory.hashState(s);
	}

	public Map<HashableState, Double> stateFrequencies(GreedyQPolicy policy) {
		final int MAX_EVAL_STEPS = 30000;
		long start = System.currentTimeMillis();
		env.resetEnvironment();
		env.setCurStateTo(initialState);

		State state = initialState;
		int numSteps = 0;
		Map<HashableState, Integer> visitCounts = new HashMap<>();
		for (State s : reachableStates) {
			visitCounts.put(h(s), 0);
		}
		while (numSteps < MAX_EVAL_STEPS) {
			visitCounts.put(h(state), visitCounts.get(h(state)) + 1);
			Action a = policy.action(state);
			EnvironmentOutcome outcome = env.executeAction(a);
			state = outcome.op;
			++numSteps;
			if (env.isInTerminalState()) {
				env.resetEnvironment();
				state = initialState;
				env.setCurStateTo(initialState);
			}
		}
		visitCounts.put(h(state), visitCounts.get(h(state)) + 1);
		Map<HashableState, Double> result = new HashMap<>();
		for (State s : reachableStates) {
			result.put(h(s), ((double) visitCounts.get(h(s))) / (double) MAX_EVAL_STEPS);
		}
		System.out.printf("stateFrequencies took %d ms\n", System.currentTimeMillis() - start);
		return result;
	}
	
	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p) {
		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, 11, 11,
				valueFunction, p);
		gui.initGUI();
	}

	private class SimpleJFrame extends JFrame {
		MultiLayerRenderer visualizer;
		int width;

		public SimpleJFrame(MultiLayerRenderer visualizer, int width) {
			this.visualizer = visualizer;
			this.width = width;
		}

		public void initGUI() {
			this.visualizer.setPreferredSize(new Dimension(width, width));
			this.visualizer.setBGColor(Color.GRAY);

			this.getContentPane().add(visualizer, BorderLayout.CENTER);
			pack();
			setVisible(true);

			this.visualizer.repaint();
		}
	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p, String outputPath) {
		if (type.equals("TrebleCross")) {
			return;
		}
		double sum = 0.0;
		double min = Double.MAX_VALUE;
		for (State s : reachableStates) {
			double v = valueFunction.value(s);
			sum += v;
			min = Math.min(min, v);
		}
		double mean = sum / reachableStates.size();
		double variance = 0.0;
		for (State s : reachableStates) {
			double v = valueFunction.value(s) - min;
			double diff = v - mean;
			variance += diff * diff;
		}
		double std = Math.sqrt(variance);
		final double minCopy = min;
		ValueFunction normalized = (s) -> (valueFunction.value(s) < -10000 ? minCopy : ((valueFunction.value(s) - mean) / std) * 10 + 3);
		
		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

		// define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation(0.3);
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		// define a 2D painter of state values,
		// specifying which attributes correspond to the x and y coordinates of the
		// canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		double scale = 1.0;
		svp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, (size)/scale), new VariableDomain(0, (size)/scale), 1, 1);
		svp.setValueStringRenderingFormat(8, Color.BLACK, 2, 0.0f, 0.75f);

		// create our ValueFunctionVisualizer that paints for all states
		// using the ValueFunction source and the state value painter we defined
		// ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates,
		// svp, valueFunction);

		// define a policy painter that uses arrow glyphs for each of the grid world
		// actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, (size)/scale), new VariableDomain(0, (size)/scale), 1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);

		MultiLayerRenderer visualizer = new MultiLayerRenderer();
		ValueFunctionRenderLayer vfLayer = new ValueFunctionRenderLayer(allStates, svp, normalized);
		PolicyRenderLayer pLayer = new PolicyRenderLayer(allStates, spp, p);

		visualizer.addRenderLayer(vfLayer);
		visualizer.addRenderLayer(pLayer);
		visualizer.setBGColor(Color.GRAY);

		SimpleJFrame frame = new SimpleJFrame(visualizer, size * 25);
		frame.setTitle(outputPath);
		frame.initGUI();

		BufferedImage image = new BufferedImage(visualizer.getWidth(), visualizer.getHeight(),
				BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics2D = image.createGraphics();
		// visualizer.repaint();
		// visualizer.paint(graphics2D);
		visualizer.repaint();
		frame.invalidate();
		// visualizer.paint(graphics2D);
		frame.paint(graphics2D);
		// frame.resize(size * 26, size * 26);
		// frame.resize(size * 25, size * 25);
		// frame.repaint();
		/*
		for (int i=0; i<5; ++i) {
			int blackCount = 0;
			for (int x = 0; x<image.getWidth(); ++x) {
				for (int y = 0; y<image.getHeight(); ++y) {
					Color c = new Color(image.getRGB(x, y));
					if (c.getBlue() < 10 && c.getRed() < 10 && c.getGreen() < 10) {
						++blackCount;
					}
				}
			}
			try {
				Thread.sleep(500);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			int numPixels = (image.getWidth() * image.getHeight());
			System.out.println("pixels = " + numPixels + ", portion = " + ((double) blackCount / numPixels));
			if (blackCount < 500) {
				break;
			}
		}
		*/
		SwingUtilities.invokeLater(() -> {
			try {
				// Sleep added to avoid graphics corruption in saved images.
				// Thread.sleep(3000); // TODO: Do something better here
				ImageIO.write(image, "png", new File(outputPath + ".png"));
				if (image.getWidth() > 250) {
					BufferedImage sub = image.getSubimage(image.getWidth() - 250, 0, 250, 250);
					ImageIO.write(sub, "png", new File(outputPath + "_subimage.png"));
				}
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
	
	private String stateActionToString(State s, Action a) {
		if (s instanceof TrebleCrossState) {
			TrebleCrossState ts = (TrebleCrossState) s;
			StringBuilder sb = new StringBuilder(ts.data);
			Integer i = Integer.parseInt(a.actionName());
			if (sb.charAt(i) == TrebleCrossDomain.X) {
				System.out.printf("PROBLEM: %d,%s,%s\n", s.hashCode(), s.toString(), a);
			}
			sb.setCharAt(i, '*');
			return sb.toString();
		}
		return s.toString();
	}
	
	private void writePolicy(GreedyQPolicy policy, String outputPath) throws IOException {
		FileWriter fw = new FileWriter(outputPath + ".txt");
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter pw = new PrintWriter(bw);
        pw.format("State,Action\n");
        Map<Integer, State> stateMap = new TreeMap<>();
		for (State s : reachableStates) {
			stateMap.put(s.hashCode(), s);
		}
		for (State s : stateMap.values()) {
			if (s.variableKeys().isEmpty()) {
				continue;
			}
			Action a = policy.action(s);
			if (s instanceof TrebleCrossState) {
				TrebleCrossState ts = (TrebleCrossState) s;
				if (ts.reward != 0) continue;
			}
	    	pw.format("%d,%s,%s\n", s.hashCode(), stateActionToString(s, a), a);
		}
        pw.close();
	}
	
	private QLearning getQLearning() {
		LearningRate learningRateFunction = new ExponentialDecayLR(learningRate, learningRateDecay, 0.001);
		QLearning qLearning = new QLearning(domain, discountFactor, hashingFactory, qInit, learningRate);
		qLearning.setLearningRateFunction(learningRateFunction);
		qLearning.setMaxQChangeForPlanningTerminaiton(MAX_QL_DELTA);
		return qLearning;
	}
	
	private GreedyQPolicy getQPolicyForViz() {
		QLearning agent = getQLearning();

		for(int i = 0; i < trialLength; i++){
			agent.runLearningEpisode(env);
			env.resetEnvironment();
		}
		
		GreedyQPolicy policy = new GreedyQPolicy(agent);
		return policy;
	}
	
	public void experimentAndPlotter(String outputFolder) {
		mkdir(outputFolder);
		String fileName = outputFolder + File.separator + String.format("%d,%.4f,%.2f,%.8f,%.2f",
				this.getNumStates(), this.discountFactor,
				this.learningRate, this.learningRateDecay, this.qInit);
		if (new File(fileName + ".csv").exists()) {
			System.out.println("Already created " + fileName);
			return;
		}
		// different reward function for more structured performance plots
		// ((FactoredModel) domain.getModel()).setRf(new GoalBasedRF(this.goalCondition,
		// 5.0, -0.1));

		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
			public String getAgentName() {
				return "Q-Learning";
			}

			public LearningAgent generateAgent() {
				return getQLearning();
			}
		};

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, NUM_TRIALS, trialLength,
				qLearningFactory);
		// LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
		// NUM_TRIALS, MAX_STEPS, qLearningFactory);
		// exp.toggleTrialLengthInterpretation(false);

		boolean doPlots = false;
		exp.toggleVisualPlots(doPlots);
		if (doPlots) {
			exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOST_RECENT_AND_AVERAGE,
					PerformanceMetric.AVERAGE_EPISODE_REWARD, PerformanceMetric.STEPS_PER_EPISODE);
		}

		exp.startExperiment();

		exp.writeEpisodeDataToCSV(fileName);
		System.out.println("Completed Q-Learning for " + fileName);
		System.gc();
	}

	private static ExperimentDomain createGridWorldDomain(int size) {
		int width = size;
		int height = size;
		GridWorldDomain gwdg = new GridWorldDomain(width, height);
		// gwdg.setMapToFourRooms();

		int split = size / 2;
		int opening1 = size / 3;
		int opening2 = size * 2 / 3;

		gwdg.verticalWall(0, opening1 - 1, split);
		gwdg.verticalWall(opening1 + 1, opening2 - 1, split);
		gwdg.verticalWall(opening2 + 1, size - 1, split);

		gwdg.horizontalWall(0, opening1 - 1, split);
		gwdg.horizontalWall(opening1 + 1, opening2 - 1, split);
		gwdg.horizontalWall(opening2 + 1, size - 1, split);

		gwdg.setRf((State s, Action a, State sprime) -> {
			GridWorldState gs = (GridWorldState) sprime;
			int reward = -1;
			if (gs.agent.x < split) {
				reward -= 1;
			}
			if (gs.agent.y < split) {
				reward -= 1;
			}
			// int reward = (width - gs.agent.x) + (height - gs.agent.y);
			if (gwdg.getTf().isTerminal(gs)) {
				reward = 4 * (size - 1) - size / 10;
				switch (size) {
				case 15: return reward - 1;
				case 20: return reward;
				case 25: return reward - 1;
				case 30: return reward + 2;
				}
				return reward;
			}
			return reward;
		});
		gwdg.setTf(new GridWorldTerminalFunction(width - 1, height - 1));

		ExperimentDomain domain = new ExperimentDomain();
		domain.type = "GridWorld";
		domain.domain = gwdg.generateDomain();
		domain.initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(width - 1, height - 1, "loc0"));
		domain.size = size;
		domain.numStates = size * size;
		return domain;
	}

	private static ExperimentDomain createTrebleCrossDomain(int size) {
		TrebleCrossDomain domainGenerator = new TrebleCrossDomain(size);
		SADomain domain = domainGenerator.generateDomain();

		ExperimentDomain result = new ExperimentDomain();
		result.type = "TrebleCross";
		result.domain = domain;
		result.initialState = domainGenerator.initialState;
		result.size = domainGenerator.numReachableStates();
		result.numStates = 1 << size;
		return result;
	}
	
	private static void writeOneIterResults(String filename, List<Double> deltas) throws IOException {
		FileWriter fw = new FileWriter(filename);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter pw = new PrintWriter(bw);
        pw.format("Iteration,Max Delta Reward\n");
        for (int i=0; i<deltas.size(); ++i) {
        	pw.format("%d,%.5f\n", i, deltas.get(i));
        }
        pw.close();
	}
	
	private static void writeIterResults(String outputFolder, List<ExperimentResults> results) throws IOException {
		mkdir(outputFolder);
		for (ExperimentResults result : results) {
			String viFilename = String.format("vi,%d,%.4f.csv", result.numStates, result.discountFactor);
			writeOneIterResults(outputFolder + File.separator + viFilename, result.viDeltaByIteration);
			if (!result.piDeltaByIteration.isEmpty()) {
				String piFilename = String.format("pi,%d,%.4f.csv", result.numStates, result.discountFactor);
				writeOneIterResults(outputFolder + File.separator + piFilename, result.piDeltaByIteration);
			}
	     }
	}
	
	private static void visualize(ExperimentConfig config) throws IOException, InterruptedException {
		System.out.println("Q-Learning Viz");
		mkdir(config.outputFolder);
		ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
		Experiment example = new Experiment(domain, config.trialLength, config.defaultDiscountFactor,
				config.defaultLearningRate, config.defaultLearningRateDecay, config.defaultQInit);
		QLearning qlAgent = example.getQLearning();
		for(int i = 0; i < config.trialLength; i++){
			qlAgent.runLearningEpisode(example.env);
			example.env.resetEnvironment();
		}
		GreedyQPolicy qlPolicy = new GreedyQPolicy(qlAgent);
		Map<HashableState, Double> freq = example.stateFrequencies(qlPolicy);
		ValueFunction freqFunc = (s) -> freq.get(example.h(s));
		
		PolicyIteration piPlanner = new PolicyIteration(domain.domain, config.defaultDiscountFactor, example.hashingFactory,
				MAX_PI_DELTA, MAX_VI_DELTA, MAX_ITERATIONS, MAX_ITERATIONS);
		GreedyQPolicy piPolicy = piPlanner.planFromState(example.initialState);
		
		ValueIteration viPlanner = new ValueIteration(domain.domain, config.defaultDiscountFactor, example.hashingFactory,
				MAX_VI_DELTA, MAX_ITERATIONS);
		GreedyQPolicy viPolicy = viPlanner.planFromState(example.initialState);
		
		if (example.type.equals("GridWorld")) {
			example.manualValueFunctionVis((ValueFunction)qlAgent, qlPolicy, config.outputFolder + File.separator + "qLearning_policy");
			example.manualValueFunctionVis(freqFunc, qlPolicy, config.outputFolder + File.separator + "qLearning_freq");
			example.manualValueFunctionVis((ValueFunction)piPlanner, piPolicy, config.outputFolder + File.separator + "pi_policy");
			example.manualValueFunctionVis((ValueFunction)viPlanner, viPolicy, config.outputFolder + File.separator + "vi_policy");
		} else {
			PrintWriter pwAll = new PrintWriter(new BufferedWriter(new FileWriter(config.outputFolder + File.separator + "all_policies.txt")));
			PrintWriter pwDiffs = new PrintWriter(new BufferedWriter(new FileWriter(config.outputFolder + File.separator + "diff_policies.txt")));
			String strFormat = String.format("%%%ds", config.defaultSize);
			String headerFormat = String.format("    # %s %s %s %%6s %%6s %%6s %%7s\n", strFormat, strFormat, strFormat);
			String rowFormat = String.format("%%4d: %s %s %s %%6.3f %%6.3f %%6.3f %%7.4f\n", strFormat, strFormat, strFormat);
	        pwAll.format(headerFormat, "VI", "PI", "QL", "VI.V", "PI.V", "QL.V", "Freq");
	        pwDiffs.format(headerFormat, "VI", "PI", "QL", "VI.V", "PI.V", "QL.V", "Freq");
	        Map<Integer, State> stateMap = new TreeMap<>();
			for (State s : example.reachableStates) {
				stateMap.put(s.hashCode(), s);
			}
			ValueFunction viv = (ValueFunction) viPlanner;
			ValueFunction piv = (ValueFunction) piPlanner;
			ValueFunction qlv = (ValueFunction) qlAgent;
			double epsilon = 0.0001;
			for (State s : stateMap.values()) {
				if (s.variableKeys().isEmpty()) {
					continue;
				}
				if (s instanceof TrebleCrossState) {
					TrebleCrossState ts = (TrebleCrossState) s;
					if (ts.reward != 0) continue;
				}
				Action viA = viPolicy.action(s);
				Action piA = piPolicy.action(s);
				Action qlA = qlPolicy.action(s);
				String viStr = example.stateActionToString(s, viA);
				String piStr = example.stateActionToString(s, piA);
				String qlStr = example.stateActionToString(s, qlA);
		    	pwAll.format(rowFormat, s.hashCode(), viStr, piStr, qlStr, viv.value(s), piv.value(s), qlv.value(s), freq.get(example.h(s)));
		    	if (Math.abs(viv.value(s) - piv.value(s)) > epsilon ||
		    			(Math.abs(viv.value(s) - qlv.value(s)) > epsilon && freq.get(example.h(s)) > 0.001)) {
		    		pwDiffs.format(rowFormat, s.hashCode(), viStr, piStr, qlStr, viv.value(s), piv.value(s), qlv.value(s), freq.get(example.h(s)));
		    	}
			}
	        pwAll.close();
	        pwDiffs.close();
		}
		/*
		String cmd = String.format("/usr/bin/soffice --convert-to png \"%sall_policies.txt\" --outdir \"%s\"",
				outputFolder, outputFolder);
		System.out.printf("Running %s\n", cmd);
		Process process = Runtime.getRuntime().exec(cmd);
		process.waitFor();
		example.writePolicy(qlPolicy, outputFolder + File.separator + "qLearning_policy");
		
		System.out.println("PI Viz");
		example.writePolicy(piPolicy, outputFolder + File.separator + "pi_policy");
		
		System.out.println("VI Viz");
		example.writePolicy(viPolicy, outputFolder + File.separator + "vi_policy");
		*/
	}

	private static void runPlanningExperiments(ExperimentConfig config)
			throws IOException, InterruptedException, ExecutionException {
		mkdir(config.outputFolder);
		String viByStateFilename = config.outputFolder + "value_by_state.csv";
		String piByStateFilename = config.outputFolder + "policy_by_state.csv";
		String viByDiscountFactorFilename = config.outputFolder + "value_by_discount_factor.csv";
		String piByDiscountFactorFilename = config.outputFolder + "policy_by_discount_factor.csv";
		if (new File(viByStateFilename).exists()
				&& new File(piByStateFilename).exists()
				&& new File(viByDiscountFactorFilename).exists()
				&& new File(piByDiscountFactorFilename).exists()) {
			System.out.println("Already ran planning for " + config.outputFolder);
			return;
		}

		List<ExperimentResults> allValueResults = new ArrayList<>();
		List<ExperimentResults> allPolicyResults = new ArrayList<>();
		ExecutorService executor = Executors.newFixedThreadPool(24);
		List<Future<Void>> futures = new ArrayList<>();

		double discountFactor = config.defaultDiscountFactor;
		double learningRate = config.defaultLearningRate;
		double learningRateDecay = config.defaultLearningRateDecay;
		double qInit = config.defaultQInit;

		// Vary size, use default choices for discount factor, learning rate, and q-init
		discountFactor = config.defaultDiscountFactor;
		learningRate = config.defaultLearningRate;
		qInit = config.defaultQInit;
		for (int size : config.sizes) {
			ExperimentDomain domain = config.factory.createDomain(size);
			Experiment example = new Experiment(domain, config.trialLength, discountFactor, learningRate, learningRateDecay, qInit);

			ExperimentResults vResults = example.valueIterationExample(config.outputFolder);
			allValueResults.add(vResults);
			System.out.println("VALUE: " + vResults);
			ExperimentResults pResults = example.policyIterationExample(config.outputFolder);
			allPolicyResults.add(pResults);
			System.out.println("POLICY: " + pResults);
		}

		ExperimentResults.writeFile(config.outputFolder + "value_by_state.csv", allValueResults);
		writeIterResults(config.outputFolder + File.separator + "vi_by_state", allValueResults);
		ExperimentResults.writeFile(config.outputFolder + "policy_by_state.csv", allPolicyResults);
		writeIterResults(config.outputFolder + File.separator + "pi_by_state", allPolicyResults);
		allValueResults.clear();
		allPolicyResults.clear();

		// Default size, vary discount factor
		for (int i = 1; i <= 19; ++i) {
			double currDiscountFactor = 1.0 - (double) i / 20.0;
			ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
			Experiment example = new Experiment(domain, config.trialLength, currDiscountFactor, learningRate, learningRateDecay, qInit);

			ExperimentResults vResults = example.valueIterationExample(config.outputFolder);
			allValueResults.add(vResults);
			System.out.println("VALUE: " + vResults);
			ExperimentResults pResults = example.policyIterationExample(config.outputFolder);
			allPolicyResults.add(pResults);
			System.out.println("POLICY: " + pResults);
		}

		ExperimentResults.writeFile(config.outputFolder + "value_by_discount_factor.csv", allValueResults);
		writeIterResults(config.outputFolder + File.separator + "vi_by_discount_factor", allValueResults);
		ExperimentResults.writeFile(config.outputFolder + "policy_by_discount_factor.csv", allPolicyResults);
		writeIterResults(config.outputFolder + File.separator + "pi_by_discount_factor", allPolicyResults);

		for (Future<Void> future : futures) {
			future.get();
		}

		executor.shutdown();
	}


	private static void runQLearningExperiments(ExperimentConfig config)
			throws IOException, InterruptedException, ExecutionException {
		mkdir(config.outputFolder);
		mkdir(config.outputFolder + File.separator + "qlearning");
		
		ExecutorService executor = Executors.newFixedThreadPool(4);
		List<Future<Void>> futures = new ArrayList<>();

		double discountFactor = config.defaultDiscountFactor;
		double learningRate = config.defaultLearningRate;
		double learningRateDecay = config.defaultLearningRateDecay;
		double qInit = config.defaultQInit;

		// Default size, discount factor. Vary learning rate.
		discountFactor = config.defaultDiscountFactor;
		for (int i = 1; i <= 19; ++i) {
			double currLearningRate = 1.0 - (double) i / 20.0;
			ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
			Experiment example = new Experiment(domain, config.trialLength, discountFactor, currLearningRate, learningRateDecay, qInit);

			Future<Void> future = executor.submit(() -> {
				System.out.println("Learning rate: " + currLearningRate);
				example.experimentAndPlotter(config.outputFolder + "qlearning/Learning Rate");
				return null;
			});
			futures.add(future);
		}

		// Default size, discount factor, learning rate. Vary qInit.
		discountFactor = config.defaultDiscountFactor;
		learningRate = config.defaultLearningRate;
		int qLimit = 35;
		for (int i = 1; i <= qLimit; ++i) {
			double currQInit = 2.0 - (double) i / 10.0;
			if (config.outputFolder.contains("gridworld_big")) {
				currQInit = 5 - i;
			}
			double finalQInit = currQInit;

			ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
			Experiment example = new Experiment(domain, config.trialLength, discountFactor, learningRate, learningRateDecay, currQInit);

			Future<Void> future = executor.submit(() -> {
				System.out.println("Q-Init: " + finalQInit);
				example.experimentAndPlotter(config.outputFolder + "qlearning/QInit");
				return null;
			});
			futures.add(future);
		}

		// Default size, discount factor, learning rate, qInit. Vary decay rate
		discountFactor = config.defaultDiscountFactor;
		learningRate = config.defaultLearningRate;
		for (int i = 7; i >= 1; --i) {
			double currLearningRateDecay = 1.0 - Math.pow(10, -i);

			ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
			Experiment example = new Experiment(domain, config.trialLength, discountFactor, learningRate, currLearningRateDecay, qInit);

			Future<Void> future = executor.submit(() -> {
				System.out.println("Decay Rate: " + currLearningRateDecay);
				example.experimentAndPlotter(config.outputFolder + "qlearning/Learning Rate Decay");
				return null;
			});
			futures.add(future);
		}

		// Vary size, use default choices for discount factor, learning rate, and q-init
		discountFactor = config.defaultDiscountFactor;
		learningRate = config.defaultLearningRate;
		qInit = config.defaultQInit;
		for (int size : config.sizes) {
			ExperimentDomain domain = config.factory.createDomain(size);
			Experiment example = new Experiment(domain, config.trialLength, discountFactor, learningRate, learningRateDecay, qInit);

			Future<Void> future = executor.submit(() -> {
				System.out.println("Size: " + size);
				example.experimentAndPlotter(config.outputFolder + "qlearning/Num States");
				return null;
			});
			futures.add(future);
		}

		// Default size, vary discount factor
		for (int i = 1; i <= 19; ++i) {
			double currDiscountFactor = 1.0 - (double) i / 20.0;
			ExperimentDomain domain = config.factory.createDomain(config.defaultSize);
			Experiment example = new Experiment(domain, config.trialLength, currDiscountFactor, learningRate, learningRateDecay, qInit);

			Future<Void> future = executor.submit(() -> {
				System.out.println("Discount factor: " + currDiscountFactor);
				example.experimentAndPlotter(config.outputFolder + "qlearning/Discount Factor");
				return null;
			});
			futures.add(future);
		}
		
		for (Future<Void> future : futures) {
			future.get();
		}

		executor.shutdown();
	}

	private static interface DomainFactory {
		ExperimentDomain createDomain(int problemSize);
	}
	
	private static class ExperimentConfig {
		String outputFolder;
		DomainFactory factory;
		int trialLength;
		List<Integer> sizes;
		int defaultSize;
		double defaultDiscountFactor;
		double defaultLearningRate;
		double defaultLearningRateDecay;
		double defaultQInit;
	}
	
	private static boolean RUN_VIZ = true;
	private static boolean RUN_PLANNING = true;
	private static boolean RUN_QL = true;
	
	private static void runSmallGridWorld(String outputFolder) throws IOException, InterruptedException, ExecutionException {
		ExperimentConfig config = new ExperimentConfig();
		config.outputFolder = outputFolder + "gridworld_small/";
		config.factory = (size) -> createGridWorldDomain(size);
		config.trialLength = 500;
		List<Integer> sizes = Arrays.asList(15, 20, 25, 30);
		config.sizes = sizes;
		config.defaultSize = 15;
		config.defaultDiscountFactor = 0.5;
		config.defaultLearningRate = 1.0;
		config.defaultLearningRateDecay = 0.99999;
		config.defaultQInit = 0.3;
		// Default discount factor was 0.99 for gridworld
		// Default learning rate was 0.1 for gridworld - this doesn't converge well. Use 0.5 or 1.0
		if (RUN_PLANNING) runPlanningExperiments(config);
		if (RUN_QL) runQLearningExperiments(config);
		if (RUN_VIZ) visualize(config);
		
		/*
		ExperimentConfig vizConfig = new ExperimentConfig();
		vizConfig.outputFolder = outputFolder + "gridworld_small/";
		vizConfig.factory = (size) -> createGridWorldDomain(size);
		vizConfig.trialLength = 500;
		vizConfig.sizes = Arrays.asList();
		vizConfig.defaultSize = 11;
		vizConfig.defaultDiscountFactor = 0.2;
		vizConfig.defaultLearningRate = 1.0;
		vizConfig.defaultLearningRateDecay = 0.99999;
		vizConfig.defaultQInit = 0.3;
		visualize(vizConfig);
		*/
	}

	private static void runLargeGridWorld(String outputFolder) throws IOException, InterruptedException, ExecutionException {
		ExperimentConfig config = new ExperimentConfig();
		config.outputFolder = outputFolder + "gridworld_big/";
		config.factory = (size) -> createGridWorldDomain(size);
		config.trialLength = 500;
		List<Integer> sizes = Arrays.asList();
		config.sizes = sizes;
		config.defaultSize = 30;
		config.defaultDiscountFactor = 0.85;
		config.defaultLearningRate = 1.0;
		config.defaultLearningRateDecay = 0.99999;
		config.defaultQInit = -5.5;
		
		if (RUN_PLANNING) runPlanningExperiments(config);
		if (RUN_QL) runQLearningExperiments(config);
		if (RUN_VIZ) visualize(config);
	}

	private static void runSmallTrebleCross(String outputFolder) throws IOException, InterruptedException, ExecutionException {
		ExperimentConfig config = new ExperimentConfig();
		config.outputFolder = outputFolder + "treblecross_small/";
		config.factory = (size) -> createTrebleCrossDomain(size);
		config.trialLength = 15000;
		List<Integer> sizes = Arrays.asList(7, 9, 11, 13);
		config.sizes = sizes;
		config.defaultSize = 7;
		config.defaultDiscountFactor = 0.7;
		config.defaultLearningRate = 1.0;
		config.defaultLearningRateDecay = 0.999;
		config.defaultQInit = 0.2;
		if (RUN_PLANNING) runPlanningExperiments(config);
		if (RUN_QL) runQLearningExperiments(config);
		if (RUN_VIZ) visualize(config);
		
		/*
		ExperimentConfig vizConfig = new ExperimentConfig();
		vizConfig.outputFolder = outputFolder + "treblecross_small/";
		vizConfig.factory = (size) -> createTrebleCrossDomain(size);
		vizConfig.trialLength = 15000;
		vizConfig.sizes = Arrays.asList();
		vizConfig.defaultSize = 6;
		vizConfig.defaultDiscountFactor = 0.7;
		vizConfig.defaultLearningRate = 1.0;
		vizConfig.defaultLearningRateDecay = 0.999;
		vizConfig.defaultQInit = 0.2;
		visualize(config);
		*/
	}

	private static void runLargeTrebleCross(String outputFolder) throws IOException, InterruptedException, ExecutionException {
		ExperimentConfig config = new ExperimentConfig();
		config.outputFolder = outputFolder + "treblecross_big/";
		config.factory = (size) -> createTrebleCrossDomain(size);
		config.trialLength = 200000;
		List<Integer> sizes = Arrays.asList();
		config.sizes = sizes;
		config.defaultSize = 13;
		config.defaultDiscountFactor = 0.99;
		config.defaultLearningRate = 1.0;
		config.defaultLearningRateDecay = 0.99999;
		config.defaultQInit = 0.0;
		
		if (RUN_PLANNING) runPlanningExperiments(config);
		if (RUN_QL) runQLearningExperiments(config);
		if (RUN_VIZ) visualize(config);
	}
	
	public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
		DPrint.toggleUniversal(false);

		String outputFolder = "/home/brandon/Dropbox/coursework/ml/project4/output/";
		if (args.length > 0) {
			outputFolder = args[1];
		}
		mkdir(outputFolder);

		runLargeTrebleCross(outputFolder);
		runLargeGridWorld(outputFolder);
		// soffice --convert-to jpg "Textfile.doc"
		
		runSmallTrebleCross(outputFolder);
		runSmallGridWorld(outputFolder);

		System.out.println("Successful completion!!!!");
	}

}