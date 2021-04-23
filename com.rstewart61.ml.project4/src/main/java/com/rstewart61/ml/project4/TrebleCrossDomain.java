package com.rstewart61.ml.project4;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

//Based on https://raw.githubusercontent.com/jmacglashan/burlap_examples/master/src/main/java/edu/brown/cs/burlap/tutorials/domain/simple/ExampleGridWorld.java

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.StatePainter;
import burlap.visualizer.StateRenderLayer;
import burlap.visualizer.Visualizer;



public class TrebleCrossDomain implements DomainGenerator {
	public static int WINDOW_SIZE = 3;
	public static char X = 'X';
	
	protected int size;
	private Map<String, Integer> positionMap = new HashMap<>();
	private Map<String, Action> actionMap = new HashMap<>();
	public TrebleCrossState[] cache;
	public Map<Integer, List<Action>> keyMap = new HashMap<>();
	public final TrebleCrossState initialState;
	private int numReachableStates = 0;

	public TrebleCrossDomain(int size) {
		this.size = size;
		cache = new TrebleCrossState[1 << size];
		initialState = new TrebleCrossState(size, cache);
	}
	
	public int numReachableStates() {
		return numReachableStates;
		/*
		int count=0;
		for (int i=0; i<cache.length; ++i) {
			if (cache[i] != null) {
				++count;
			}
		}
		return count;
		*/
	}
	
	private ActionType makeActionType(String actionName) {
		SimpleAction action = new SimpleAction(actionName);
		return new ActionType() {
			@Override
			public String typeName() {
				return actionName;
			}

			@Override
			public Action associatedAction(String strRep) {
				return action;
			}

			@Override
			public List<Action> allApplicableActions(State s) {
				int hash = s.hashCode();
				if (!keyMap.containsKey(hash)) {
					List<Action> result = new ArrayList<>();
					for (Object key : s.variableKeys()) {
						result.add(actionMap.get(key.toString()));
					}
					keyMap.put(hash, result);
				}
				return keyMap.get(hash);
			}
			
		};
	}

	@Override
	public SADomain generateDomain() {
		SADomain domain = new SADomain();

		for (int i=0; i<size; ++i) {
			String actionName = "" + i;
			positionMap.put(actionName, i);
			actionMap.put(actionName, new SimpleAction(actionName));
			domain.addActionType(makeActionType(actionName));
		}

		TrebleCrossStateModel smodel = new TrebleCrossStateModel();
		RewardFunction rf = new TrebleCrossRF();
		TerminalFunction tf = new TrebleCrossTF();

		domain.setModel(new FactoredModel(smodel, rf, tf));

		mapState(initialState, 0);
		// System.out.println("Num reachable states for size " + size + " is " + numReachableStates);

		return domain;
	}
	
	private void mapState(TrebleCrossState state, int player) {
		List<Object> keys = state.variableKeys();
		for (Object key : keys) {
			int nextHash = state.getNextHash(key);
			if (cache[nextHash] == null) {
				++numReachableStates;
				TrebleCrossState nextState = state.getNextState(key);
				cache[nextHash] = nextState; // redundant
				boolean isLoser = nextState.loses();
				if (isLoser) {
					switch (player) {
					case 0:
						nextState.reward = -1;
						break;
					case 1:
						nextState.reward = 1;
						break;
					}
				} else {
					mapState(nextState, (player + 1) % 2);
				}
				nextState.updateKeys();
				// System.out.println("Mapped " + nextState.toString());
			}
		}
	}


	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStatePainter(new TrebleCrossDomain.AgentPainter());


		return rl;
	}

	public Visualizer getVisualizer(){
		return new Visualizer(this.getStateRenderLayer());
	}


	private int getPosition(Action a) {
		return positionMap.get(a.actionName());
	}

	Random random = new Random();
	protected class TrebleCrossStateModel implements FullStateModel{
		public TrebleCrossStateModel() {
		}

		@Override
		public List<StateTransitionProb> stateTransitions(State s, Action a) {
			TrebleCrossState gs = (TrebleCrossState)s;
			if (gs.reward != 0) {
				return Arrays.asList(new StateTransitionProb(s, 1.0));
			}

			TrebleCrossState ns = gs.getNextState(getPosition(a));
			List<Object> keys = ns.variableKeys();
			List<StateTransitionProb> tps = new ArrayList<StateTransitionProb>(keys.size());
			if (ns.reward != 0) {
				// player loses
				tps.add(new StateTransitionProb(ns, 1.0));
			} else {
				for (Object key : keys) {
					TrebleCrossState nns = ns.getNextState(key);
					tps.add(new StateTransitionProb(nns, 1.0 / keys.size()));
				}
			}

			return tps;
		}

		@Override
		public State sample(State s, Action a) {
			TrebleCrossState gs = (TrebleCrossState)s;
			TrebleCrossState ns = gs.getNextState(getPosition(a));
			if (ns.reward != 0) {
				return ns;
			}
			
			List<Object> keys = ns.variableKeys();
			int pos = random.nextInt(keys.size());
			Object sampledKey = keys.get(pos);
			TrebleCrossState nns = ns.getNextState(sampledKey);

			return nns;
		}
	}

	public class AgentPainter implements StatePainter {

		@Override
		public void paint(Graphics2D g2, State s,
						  float cWidth, float cHeight) {
			System.out.println(s.toString());
			System.out.println(s.variableKeys());
			g2.setColor(Color.BLACK);
			TrebleCrossState ts = (TrebleCrossState) s;
			if (ts.reward > 0) {
				g2.setColor(Color.GREEN);
			} else if (ts.reward < 0) {
				g2.setColor(Color.RED);
			}

			//set up floats for the width and height of our domain
			float fWidth = size;
			float fHeight = 1;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			//top coordinate of cell on our canvas
			//coordinate system adjustment because the java canvas
			//origin is in the top left instead of the bottom right
			float ry = cHeight - height;

			for (int i=0; i<size; ++i) {
				Character value = (Character)s.get(i);
				
				//left coordinate of cell on our canvas
				float rx = i*width;

				if (value.equals(X)) {
					g2.draw(new Line2D.Float(rx, ry, rx+width, ry+height));
					g2.draw(new Line2D.Float(rx+width, ry, rx, ry+height));
				}
			}
		}
	}

	public class TrebleCrossRF implements RewardFunction {
		@Override
		public double reward(State s, Action a, State sprime) {
			return ((TrebleCrossState) sprime).reward;
			/*
			// System.out.print("Action " + a + " ");
			if (sprime == TrebleCrossState.WIN) {
				// System.out.println("WIN with: " + s + " -> " + sprime);
				return 1.0;
			}
			if (sprime == TrebleCrossState.LOSE) {
				// System.out.println("LOSE with: " + s + " -> " + sprime);
				return -1.0;
			}
			// System.out.println("CONTINUE with: " + s + " -> " + sprime);
			return 0.0;
			*/
		}
	}

	public class TrebleCrossTF implements TerminalFunction {
		@Override
		public boolean isTerminal(State s) {
			return ((TrebleCrossState) s).reward != 0;
		}
	}

	public static void main(String [] args) {
		int boardSize = 10;
		TrebleCrossDomain gen = new TrebleCrossDomain(boardSize);
		SADomain domain = gen.generateDomain();
		SimulatedEnvironment env = new SimulatedEnvironment(domain, gen.initialState);

		Visualizer v = gen.getVisualizer();
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		for (int i=0; i<boardSize; ++i) {
			exp.addKeyAction("" + i, "" + i, "");
		}

		exp.initGUI();
	}

}